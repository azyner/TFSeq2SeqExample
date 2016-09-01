import tensorflow as tf
import numpy as np
import random
import data_utils
from tensorflow.python.ops import nn_ops


class Seq2SeqModel(object):

    def __init__(self, parameters, feed_forward, train, encoder_steps, decoder_steps, batch_size,
                 rnn_size, num_layers,learning_rate,learning_rate_decay_factor, input_size, max_gradient_norm):
        # feed_forward: whether or not to use a loopback function and therefore feed the last ouput
        #                to the next input during sequence generation
        # train: train the model (or test)

        self.max_gradient_norm = max_gradient_norm
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        dtype = tf.float32

        self.batch_size = batch_size
        self.input_size = input_size
        self.encoder_steps = encoder_steps
        self.decoder_steps = decoder_steps
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
        self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        if feed_forward and not train:
            print "Warning, feeding the model future sequence data (feed_forward) is not recommended when the model is not training."

        # The output of the multiRNN is the size of rnn_size, and it needs to match the input size, or loopback makes
        #  no sense. Here a single layer without activation function is used, but it can be any number of
        #  non RNN layers / functions
        w = tf.get_variable("proj_w", [self.rnn_size, self.input_size])
        b = tf.get_variable("proj_b", [self.input_size])
        output_projection = (w, b)

        # define layers here
        # input, linear RNN RNN linear etc

        # Default should be True, but TF 0.9 was throwing a warning, implying it was false
        single_cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size,state_is_tuple=True)
        cell = single_cell
        if self.num_layers > 1:
            # state_is_tuple defaults to False in TF0.9, and thus produces a warning....
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * self.num_layers,state_is_tuple=True)

        def simple_loop_function(prev, _):
            '''Function that takes last output, and applies output projection to it'''
            if output_projection is not None:
                prev = nn_ops.xw_plus_b(
                        prev, output_projection[0], output_projection[1])
            return prev

        from tensorflow.python.ops import variable_scope
        from tensorflow.python.framework import dtypes
        from tensorflow.python.ops import rnn
        from tensorflow.python.ops.seq2seq import rnn_decoder

        #TODO move these functions to their own file.
        def basic_rnn_seq2seq_with_loop_function(
                encoder_inputs, decoder_inputs, cell, dtype=dtypes.float32,loop_function=simple_loop_function,scope=None):
            """Basic RNN sequence-to-sequence model. Edited for a loopback function. Don't know why this isn't in the
            current library
            """
            with variable_scope.variable_scope(scope or "basic_rnn_seq2seq_with_loop_function"):
                _, enc_state = rnn.rnn(cell, encoder_inputs, dtype=dtype)
                return rnn_decoder(decoder_inputs, enc_state, cell,loop_function=loop_function)

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs, feed_forward):
            if not feed_forward: #feed last output as next input
                loopback_function = simple_loop_function
            else:
                loopback_function = None #feed correct input
            return basic_rnn_seq2seq_with_loop_function(encoder_inputs,decoder_inputs,cell,
                                                                      loop_function=loopback_function,dtype=dtype)

        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in xrange(self.encoder_steps):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.float32, shape=[batch_size, 1],
                                                    name="encoder{0}".format(i)))
        for i in xrange(self.decoder_steps + 1):
            self.decoder_inputs.append(tf.placeholder(tf.float32, shape=[batch_size, 1],
                                                    name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(dtype, shape=[batch_size],
                                                    name="weight{0}".format(i)))

        # Our targets are decoder inputs shifted by one.
        # TODO Alex Double check this
        # PLACEHOLDER HACK - because I don't believe <go> or <eos> symbols belong in time series data, I am repeating
        # the last decoder input for the target until I replace it with something better.
        # I think the correct solution here is to actually capture the last ignored output, and use it as the
        # first decoder symbol instead of the <go> symbol.
        # I'm not sure I'm allowed to tie the unused output to the decoder feed, so I'm doing this for now.

        targets = [self.decoder_inputs[i + 1]
                    for i in xrange(len(self.decoder_inputs) - 1)]
        targets.append(self.decoder_inputs[len(self.decoder_inputs)-1])

        if train: #Training
            self.outputs, self.internal_states = seq2seq_f(self.encoder_inputs, self.decoder_inputs, feed_forward)
        else: #Testing
            self.outputs, self.internal_states = seq2seq_f(self.encoder_inputs, self.decoder_inputs, feed_forward)

        # self.outputs is a list of len(decoder_steps+1) containing [size batch x rnn_size]
        # The output projection below reduces this to:
        #                 a list of len(decoder_steps+1) containing [size batch x input_size]
        if output_projection is not None:
            self.outputs = [
                    nn_ops.xw_plus_b(output, output_projection[0], output_projection[1])
                    for output in self.outputs
                ]

        def rmse(x, y):
            return tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y, x))))

        # TODO There are several types of cost functions to compare tracks. Implement many
        # Mainly, average MSE over the whole track, or just at a horizon time (t+10 or something)
        # There's this corner alg that Social LSTM refernces, but I haven't looked into it.

        self.losses = tf.nn.seq2seq.sequence_loss(self.outputs,targets,self.target_weights,softmax_loss_function=lambda x, y: rmse(x,y))

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if train:
            self.gradient_norms = []
            self.updates = []
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            gradients = tf.gradients(self.losses, params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)

            self.gradient_norms.append(norm)
            self.updates.append(opt.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step))

        self.saver = tf.train.Saver(tf.all_variables())

        tf.scalar_summary('Loss',self.losses)

    def get_batch(self, encoder_data, decoder_data):
        # This whole function just collects random pairs of encoder/decoder from data and adds them into a batch
        # This is where the target weight is created, it is zero for padding, 1 for everything else
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
        encoder_inputs, decoder_inputs = [], []

        # Get a random batch of encoder and decoder inputs from data, add GO to decoder.
        for _ in xrange(self.batch_size):
            index = random.randrange(encoder_data.shape[0])
            encoder_input = encoder_data[index]
            decoder_input = decoder_data[index]
            encoder_inputs.append(encoder_input)

            # Decoder inputs get an extra "GO" symbol
            # The fact that decoder input is a ndarray and not a list breaks this operator (+), so it is recast tolist
            decoder_inputs.append([[data_utils.GO_ID]*self.input_size] + decoder_input.tolist())

        # Batch encoder inputs are just re-indexed encoder_inputs.
        # Need to re-index to make an encoder_steps long list of shape [batch input_size]
        # currently it is a list of length batch containing shape [timesteps input_size]

        for length_idx in xrange(self.encoder_steps):
            batch_encoder_inputs.append(
                    np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.float32))

        for length_idx in xrange(self.decoder_steps+1): # +1 for go symbol
            batch_decoder_inputs.append(
                    np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.float32))

            # Because of  the offset with the GO symbol, the final target is a repeat of the second last target
            # It is therefore given weight zero, as it is not important
            # All other targets are equally important, so they are weighted as 1.0
            # Alex - This could be reason for the random decay I am observing
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in xrange(self.batch_size):
                if length_idx == self.decoder_steps:
                  batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)

        # Batch_encoder_inputs is now list of len encoder_steps, shape batch, input_size. Similarly with decoder_inputs
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights

    def step(self, session, encoder_inputs, decoder_inputs, target_weights,
             bucket_id, feed_forward, train_model, summary_writer=None):
        """Run a step of the model feeding the given inputs.
        Args:
          session: tensorflow session to use.
          encoder_inputs: list of numpy int vectors to feed as encoder inputs.
          decoder_inputs: list of numpy int vectors to feed as decoder inputs.
          target_weights: list of numpy float vectors to feed as target weights.
          bucket_id: which bucket of the model to use.
          train: whether to do the backward step or only forward.
        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.
        Raises:
          ValueError: if length of encoder_inputs, decoder_inputs, or
            target_weights disagrees with bucket size for the specified bucket_id.
        """

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in xrange(self.encoder_steps):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(self.decoder_steps+1):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[self.decoder_steps].name
        input_feed[last_target] = np.array([np.zeros(self.input_size,dtype=np.float32)]*self.batch_size)

        # Output feed: depends on whether we do a backward step or not.
        if train_model:
            output_feed = (self.updates +  # Update Op that does SGD. #This is the learning flag
                         self.gradient_norms +  # Gradient norm.
                         [self.losses])  # Loss for this batch.
        else:
            #This whole ouput format is really bad form, it makes adding a tensorboard summary difficult as
            #different variables (loss, output,etc) share the same name.
            output_feed = [self.losses]  # Loss for this batch.
            for l in xrange(self.decoder_steps+1):  # Output logits.
                output_feed.append(self.outputs[l])


        outputs = session.run(output_feed, input_feed)
        if summary_writer is not None:
            summary_op = tf.merge_all_summaries()
            summary_str = session.run(summary_op,input_feed)
            summary_writer.add_summary(summary_str, self.global_step.eval())
        if train_model:
            return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
        else:
            return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.
