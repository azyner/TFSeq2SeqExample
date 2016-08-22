#load params

#sort data into feed_dicts

#training loop, report losses, etc

#declare session, checkpoints, loading etc






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
    # Alex -- Plenty of documentation in seq2seq
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
    # ?????????????
    lstm_state_size = np.sum(parameters['lstm_layers']) * 2
    istate = tf.placeholder(tf.float32, shape=(None, lstm_state_size), name='internal_state')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    input_size = 1  ##PLACEHOLDER

    RNNInputSize = parameters['n_input']

    return -1


def training():
    return -1


def generate():
    # Is this just a decoder wrapper?
    return -1
