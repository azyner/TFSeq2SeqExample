import tensorflow as tf
import numpy as np


class Seq2SeqModel(object):

    def __init__(self,parameters):

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

        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
          self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[batch_size],
                                                    name="encoder{0}".format(i)))
        for i in xrange(buckets[-1][1] + 1):
          self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[batch_size],
                                                    name="decoder{0}".format(i)))
          self.target_weights.append(tf.placeholder(dtype, shape=[batch_size],
                                                    name="weight{0}".format(i)))

        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i + 1]
                   for i in xrange(len(self.decoder_inputs) - 1)]

        # Training outputs and losses.
        if forward_only: #Test
          self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
              self.encoder_inputs, self.decoder_inputs, targets,
              self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
              softmax_loss_function=softmax_loss_function)
          # If we use output projection, we need to project outputs for decoding.
          if output_projection is not None:
            for b in xrange(len(buckets)):
              self.outputs[b] = [
                  tf.matmul(output, output_projection[0]) + output_projection[1]
                  for output in self.outputs[b]
              ]
        else: #Training
          self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
              self.encoder_inputs, self.decoder_inputs, targets,
              self.target_weights, buckets,
              lambda x, y: seq2seq_f(x, y, False),
              softmax_loss_function=softmax_loss_function)

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if not forward_only:
          self.gradient_norms = []
          self.updates = []
          opt = tf.train.GradientDescentOptimizer(self.learning_rate)
          for b in xrange(len(buckets)):
            gradients = tf.gradients(self.losses[b], params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                             max_gradient_norm)
            self.gradient_norms.append(norm)
            self.updates.append(opt.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step))

        self.saver = tf.train.Saver(tf.all_variables())

        return -1

    #TODO There are several types of cost functions to compare tracks. Implement many
    #Mainly, average MSE over the whole track, or just at a horizon time (t+10 or something)
    def sequence_loss_by_example_linear():
        #The original sequence_loss_by_example assumes logits and is probably for word processing
        #Or, I could just insert a MSE loss in the crossentropy loss function. Easier.


        #To follow the same structure:

        """Weighted loss for a sequence of logits, per-example
            Args:
                logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
                targets: List of 1D batch-sized int32 Tensors of the same length as logits.
                weights: List of 1D batch-sized float-Tensors of the same length as logits.
                    average_across_timesteps: If set, divide the returned cost by the total
                    label weight.
                    average_across_batch: If set, divide the returned cost by the batch size.
                    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
                    to be used instead of the standard softmax (the default if this is None).
                name: Optional name for this operation, defaults to "sequence_loss".
            Returns:
                A scalar float Tensor: The average log-perplexity per symbol (weighted).
            Raises:
                ValueError: If len(logits) is different from len(targets) or len(weights).
        """
        return -1
    def loss():

        return -1


    # Returns the LSTM stack created based on the parameters.
    # Processes several batches at once.
    # Input shape is: (parameters['batch_size'], parameters['n_steps'], parameters['n_input'])
    def RNN(parameters, input, model, initial_state):
        # The model is:
        # 1. input
        # 2. linear layer
        # 3 - n. LSTM layers
        # n+1. linear layer
        # n+1. output

        # 1. layer, linear activation for each batch and step.
        if (model.has_key('input_weights')):
            input = tf.matmul(input, model['input_weights']) + model['input_bias']
            # input = tf.nn.dropout(input, model['keep_prob'])

        # # Split data because rnn cell needs a list of inputs for the RNN inner loop,
        # # that is, a n_steps length list of tensors shaped: (batch_size, n_inputs)
        #Alex -- Plenty of documentation in seq2seq
        # # This is not well documented, but check for yourself here: https://goo.gl/NzA5pX
        # input = tf.split(0, parameters['n_steps'], input)  # n_steps * (batch_size, :)

        # Note: States is shaped: batch_size x cell.state_size
        outputs, states = tf.nn.rnn(model['rnn_cell'], input, initial_state=initial_state)


        # outputs[-1] = tf.Print(outputs[-1], [outputs[-1]], "LSTM Output: ", summarize = 100)
        lastOutput = tf.verify_tensor_all_finite(outputs[-1], "LSTM Outputs not finite!")


        # lastOutput = tf.nn.dropout(lastOutput, model['keep_prob'])
        # Only the last output is interesting for error back propagation and prediction.
        # Note that all batches are handled together here.
        raw_output = tf.matmul(lastOutput, model['output_weights']) + model['output_bias']
        raw_output = tf.verify_tensor_all_finite(raw_output, "Raw output not finite!")

        # And now, instead of just outputting the expected value, we output mixture distributions.
        # The number of mixtures is intuitively the number of possible actions the target can take.
        # The output is divided into triplets of n_mixtures mixture parameters for the 2 absolute position coordinates.
        if parameters['MDN']:
            output = softmax_mixtures(raw_output, n_mixtures, batch_size)
        else:
            output = raw_output
        # output = tf.Print(output, [output], "Output: ", summarize = 100)
        output = tf.verify_tensor_all_finite(output, "Final output not finite!")

        return (output, states)


    def create(parameters):
        print "Creating the neural network graph."
        tf.reset_default_graph()

        x = tf.placeholder(tf.float32, shape=(None, parameters['n_steps'], parameters['n_input']), name='input')
        y = tf.placeholder(tf.float32, shape=(None, parameters['n_output']), name='expected_output')
        #?????????????
        lstm_state_size = np.sum(parameters['lstm_layers']) * 2
        istate = tf.placeholder(tf.float32, shape=(None, lstm_state_size), name='internal_state')
        lr = tf.placeholder(tf.float32, name='learning_rate')
        input_size = 1 ##PLACEHOLDER

        RNNInputSize = parameters['n_input']



        return -1


    def training():
        return -1

    def generate():
        #Is this just a decoder wrapper?
        return -1