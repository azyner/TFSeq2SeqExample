import tensorflow as tf
import numpy as np
import random
from six.moves import xrange  # pylint: disable=redefined-builtin

class Seq2SeqModel(object):

    def __init__(self, parameters, generate, encoder_steps, decoder_steps, batch_size):
        max_gradient_norm = 5.0
        size = 11
        num_layers = 3
        dtype = tf.float32
        self.batch_size = batch_size
        self.input_size = 1
        self.encoder_steps = encoder_steps
        self.decoder_steps = decoder_steps

        #TODO
        #killall:
        # buckets - used for different length strings, sort into buckets to minimize badding but keep enough for a batch
        # output_projection - used to project from a smaller word2vec to a full vocab vector. Useless here
        #
        #Understand:
        #forward only. What is forward only? Data use? Should I disable during training, and enable during validation/generation?


        #define layers here
        #input, linear RNN RNN linear etc

        single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
        cell = single_cell
        if num_layers > 1:
          cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return tf.nn.seq2seq.basic_rnn_seq2seq(encoder_inputs,decoder_inputs,cell,dtype=dtype)
            #basic_rnn_seq2seq returns rnn_decoder returns output, state
            #there is no loss function here
          # return tf.nn.seq2seq.embedding_attention_seq2seq(
          #     encoder_inputs,
          #     decoder_inputs,
          #     cell,
          #     num_encoder_symbols=source_vocab_size,
          #     num_decoder_symbols=target_vocab_size,
          #     embedding_size=size,
          #     output_projection=output_projection,
          #     feed_previous=do_decode,
          #     dtype=dtype)

        # Feeds for input - no longer a list of buckets, just one will do.
            #NOTE Weights:
                #This is a list of weights used to determine the most important (favourable?) output. It weights the loss function
                #Used in sequence_loss_by_example.
        #NOTE - DECODERS:
            # They are the correct prediction path. Useful to correct errors in training
            # Need to do more research in the architecture here

        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in xrange(self.encoder_steps):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.float32, shape=[batch_size, 1],
                                                    name="encoder{0}".format(i)))
        #for i in xrange(self.decoder_steps + 1):
        for i in xrange(self.decoder_steps):
            self.decoder_inputs.append(tf.placeholder(tf.float32, shape=[batch_size, 1],
                                                    name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(dtype, shape=[batch_size],
                                                    name="weight{0}".format(i)))


        #self.target = tf.placeholder(tf.float32,shape=[batch_size, self.input_size],name="target")

        # Our targets are decoder inputs shifted by one.
        #TODO Alex Double check this
        #PLACEHOLDER HACK - because I don't believe <go> or <eos> symbols belong in time series data, I am repeating the
        #last decoder input for the target until I replace it with something better.
        #I think the correct solution here is to actually capture the last ignored output, and use it as the first decoder symbol
        #instead of the <go> symbol.
        #I'm not sure I'm allowed to tie the unused output to the decoder feed, so I'm doing this for now.

        targets = [self.decoder_inputs[i + 1]
                    for i in xrange(len(self.decoder_inputs) - 1)]
        #targets.append(tf.placeholder(tf.float32, shape=[batch_size, 1], name="decoder{0}".format(self.decoder_steps+1)))
        targets.append(self.decoder_inputs[len(self.decoder_inputs)-1])
        # Alex - I don't know what the difference is between forward only and not. (refactored to generate)
        # Training outputs and losses.
        #seq2seq_f: encoder_inputs is a LIST of 2D tensors of size [batch x input_size]
        #The comments on seq2seq seem to imply that the list should be timesteps long
        if generate: #Test
            self.outputs, self.internal_states = seq2seq_f(self.encoder_inputs, self.decoder_inputs, generate)
        else: #Training
            self.outputs, self.internal_states = seq2seq_f(self.encoder_inputs, self.decoder_inputs, generate)

        def RMSE(x,y):
            return  tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y, x))))
        #weights = np.ones(len(self.target))#list of 1's the size of self.target

        # TODO There are several types of cost functions to compare tracks. Implement many
        # Mainly, average MSE over the whole track, or just at a horizon time (t+10 or something)
        # There's this corner alg that Social LSTM refernces, but I haven't looked into it.

        self.losses = tf.nn.seq2seq.sequence_loss_by_example(self.outputs,targets,self.target_weights,softmax_loss_function=lambda x, y: RMSE(x,y))

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if not generate:
            self.gradient_norms = []
            self.updates = []
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)

            self.gradient_norms.append(norm)
            self.updates.append(opt.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step))

        self.saver = tf.train.Saver(tf.all_variables())

    def get_batch(self, encoder_data, decoder_data):
        #This whole function just collects random pairs of encoder/decoder from data and adds them into a batch
        #This is where the target weight is created, it is zero for padding, 1 for everything else
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
            #encoder_size, decoder_size = #####GLOBAL TRACK INPUT AND OUTPUT SIZE
        encoder_inputs, decoder_inputs = [], []

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in xrange(self.batch_size):
          #encoder_input, decoder_input = random.choice()
          index = random.randrange(encoder_data.shape(0))
          encoder_input = encoder_data[index]
          decoder_input = decoder_data[index]

          #encoder_data.size 2538 23 1
          #decoder_data.size 2538 31 1

          #Pick random int from encoder_data.size[0]

          # Encoder inputs are padded and then reversed. ### ALEX ### -- HUH? Why reversed?
          #Language works better in reverse -- dont ask

          encoder_inputs.append(encoder_input)

          # Decoder inputs get an extra "GO" symbol
          decoder_inputs.append([data_utils.GO_ID] + decoder_input)

        # Batch encoder inputs are just re-indexed encoder_inputs.
        #TODO Alex -- how are the data re-indexed? This is convoluted and not commented
        #It appears to be an unroll of the batch, does it swap axes?
        
        for length_idx in xrange(len(encoder_inputs)):
          batch_encoder_inputs.append(
              np.array([encoder_inputs[batch_idx][length_idx]
                        for batch_idx in xrange(self.batch_size)], dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
          batch_decoder_inputs.append(
              np.array([decoder_inputs[batch_idx][length_idx]
                        for batch_idx in xrange(self.batch_size)], dtype=np.int32))

          # Create target_weights to be 0 for targets that are padding.
          batch_weight = np.ones(self.batch_size, dtype=np.float32)
          for batch_idx in xrange(self.batch_size):
            # We set weight to 0 if the corresponding target is a PAD symbol.
            # The corresponding target is decoder_input shifted by 1 forward.
            if length_idx < decoder_size - 1:
              target = decoder_inputs[batch_idx][length_idx + 1]
            if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
              batch_weight[batch_idx] = 0.0
          batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights

    def step(self, session, encoder_inputs, decoder_inputs, target_weights,
               bucket_id, forward_only):
        """Run a step of the model feeding the given inputs.
        Args:
          session: tensorflow session to use.
          encoder_inputs: list of numpy int vectors to feed as encoder inputs.
          decoder_inputs: list of numpy int vectors to feed as decoder inputs.
          target_weights: list of numpy float vectors to feed as target weights.
          bucket_id: which bucket of the model to use.
          forward_only: whether to do the backward step or only forward.
        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.
        Raises:
          ValueError: if length of encoder_inputs, decoder_inputs, or
            target_weights disagrees with bucket size for the specified bucket_id.
        """
        # Check if the sizes match.
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
          raise ValueError("Encoder length must be equal to the one in bucket,"
                           " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
          raise ValueError("Decoder length must be equal to the one in bucket,"
                           " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
          raise ValueError("Weights length must be equal to the one in bucket,"
                           " %d != %d." % (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in xrange(encoder_size):
          input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
          input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
          input_feed[self.target_weights[l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
          output_feed = [self.updates,  # Update Op that does SGD.
                         self.gradient_norms,  # Gradient norm.
                         self.losses]  # Loss for this batch.
        else:
          output_feed = [self.losses]  # Loss for this batch.
          for l in xrange(decoder_size):  # Output logits.
            output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
          return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
        else:
          return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.
