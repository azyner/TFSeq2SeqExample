import tensorflow as tf
import numpy as np


class Seq2SeqModel(object):

    def __init__(self, parameters, generate):
        max_gradient_norm = 5.0
        size = 10
        num_layers = 2
        dtype = tf.float32
        batch_size = 17

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

        self.encoder_input = tf.placeholder(tf.int32, shape=[batch_size], name="encoder")
        self.decoder_input = tf.placeholder(tf.int32, shape=[batch_size], name="decoder")
        self.target_weight = tf.placeholder(tf.int32, shape=[batch_size], name="target_weight")
        self.target = tf.placeholder(tf.int32,shape=[batch_size],name="target")
        # Our targets are decoder inputs shifted by one.
        #Alex - True for my model as well
        targets = [self.decoder_inputs[i + 1]
                   for i in xrange(len(self.decoder_inputs) - 1)]

        # Alex - I don't know what the difference is between forward only and not. (refactored to generate)
        # Training outputs and losses.
        if generate: #Test
            self.output, self.internal_state = seq2seq_f(self.encoder_inputs, self.decoder_inputs, generate)
        else: #Training
            self.output, self.internal_state = seq2seq_f(self.encoder_inputs, self.decoder_inputs, generate)

        def RMSE(x,y):
            return  tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y, x))))
        weights = np.ones(len(self.target))#list of 1's the size of self.target

        # TODO There are several types of cost functions to compare tracks. Implement many
        # Mainly, average MSE over the whole track, or just at a horizon time (t+10 or something)
        # There's this corner alg that Social LSTM refernces, but I haven't looked into it.

        self.loss = tf.nn.seq2seq.sequence_loss_by_example(self.output,self.target,weights,softmax_loss_function=lambda x, y: RMSE(x,y))

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


