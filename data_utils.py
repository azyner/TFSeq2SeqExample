
# Special vocabulary symbols. Artifact from the vocab system. I don't know a good way to replace this in a linear system

_PAD = 0.0
_GO = 999
#_EOS = b"_EOS"
#_UNK = b"_UNK"
#_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0.0
GO_ID = -2.0
EOS_ID = 2.0
UNK_ID = 3.0


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn


import logging
logging.basicConfig(level=logging.INFO)

def x_sin(x):
    return x * np.sin(x)

def sin_cos(x):
    return pd.DataFrame(dict(a=np.sin(x), b=np.cos(x)), index=x)

def rnn_data(data, encoder_steps, decoder_steps):
    """
    creates new data frame based on previous observation
      * example:
        l = [1, 2, 3, 4, 5]
        time_steps = 2
        -> labels == False [[1, 2], [2, 3], [3, 4]]
        -> labels == True [2, 3, 4, 5]
    """
    rnn_df_encoder = []
    rnn_df_decoder = []
    #TODO ALEX Change range, return pair of data, encoder_data, decoder_data
    for i in range(len(data) - (encoder_steps+decoder_steps)):
        try:
            rnn_df_decoder.append(data.iloc[i + encoder_steps:i +(encoder_steps+decoder_steps)].as_matrix())
        except AttributeError:
            rnn_df_decoder.append(data.iloc[i + encoder_steps:i +(encoder_steps+decoder_steps)])
        data_ = data.iloc[i: i + encoder_steps].as_matrix()
        rnn_df_encoder.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])
    return np.array(rnn_df_encoder), np.array(rnn_df_decoder)


def split_data(data, val_size=0.1, test_size=0.1):
    """
    splits data to training, validation and testing parts
    """
    ntest = int(round(len(data) * (1 - test_size)))
    nval = int(round(len(data.iloc[:ntest]) * (1 - val_size)))

    df_train, df_val, df_test = data.iloc[:nval], data.iloc[nval:ntest], data.iloc[ntest:]

    return df_train, df_val, df_test

def prepare_data(data, encoder_steps, decoder_steps, labels=False, val_size=0.1, test_size=0.1):
    """
    Given the number of `time_steps` and some data,
    prepares training, validation and test data for an lstm cell.
    """
    df_train, df_val, df_test = split_data(data, val_size, test_size)
    return (rnn_data(df_train, encoder_steps, decoder_steps),
            rnn_data(df_val, encoder_steps, decoder_steps),
            rnn_data(df_test, encoder_steps, decoder_steps))

def generate_data(fct, x, fct_mod, encoder_steps, decoder_steps, seperate=False):

    """generates data with based on a function fct
    input:
    fct: The function to be used to generate data (eg sin)
    x: the linspace to pass to the function
    fct mod: A list of elements of 4 tuples that represent function modifiers: a+b*fct(c+d*x)
    """

    train_x, val_x, test_x = [],[],[]
    train_y, val_y, test_y = [],[],[]

    for wave in fct_mod:
        a = wave[0]
        b = wave[1]
        c = wave[2]
        d = wave[3]
        data = a+b*fct(c+d*x)
        #If there is only 1 function, do the regular split for training /test /val
        if(len(fct_mod)==1):
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            w_train, w_val, w_test = prepare_data(data['a'] if seperate else data, encoder_steps, decoder_steps)
            train_x.extend(w_train[0])
            val_x.extend(w_val[0])
            test_x.extend(w_test[0])
            train_y.extend(w_train[1])
            val_y.extend(w_val[1])
            test_y.extend(w_test[1])
        else:
            #training / val are most of data. Test is the last function.
            if(wave is not fct_mod[-1]):
                if not isinstance(data, pd.DataFrame):
                    data = pd.DataFrame(data)
                w_train, w_val, w_test = prepare_data(data['a'] if seperate else data, encoder_steps, decoder_steps, test_size = 0)
                train_x.extend(w_train[0])
                val_x.extend(w_val[0])
                test_x.extend(w_test[0])
                train_y.extend(w_train[1])
                val_y.extend(w_val[1])
                test_y.extend(w_test[1])
            else:
                #last function track, use for training
                if not isinstance(data, pd.DataFrame):
                    data = pd.DataFrame(data)
                test_x, test_y = rnn_data(data, encoder_steps, decoder_steps)

    return dict(train=np.array(train_x),
                val=np.array(val_x),
                test=np.array(test_x)), \
           dict(train=np.array(train_y),
                val=np.array(val_y),
                test=np.array(test_y))

def generate_sequence(regressor, test_sequence, seed_timesteps, prediction_length=None):
    if prediction_length > len(test_sequence)-seed_timesteps:
        raise AssertionError("Prediction length must be less than len(test_sequence)-seed_timesteps")
    if prediction_length == None:
        prediction_length = len(test_sequence)-seed_timesteps
    track = test_sequence[0:seed_timesteps]
    for i in range(prediction_length):
        packed =np.array([track])
        temp = regressor.predict(packed,axis=2)
        track = np.insert(track,track.shape[0],temp,axis=0) #Insert used (not append) to prevent array of shape (T,1)
                                                            # collapsing to a 1D array of (T,)
        # track = np.append(track,temp)
        #print track
        print len(track)
        #track = np.concatenate([track,regressor.predict(track)])
    return track

if __name__ == "__main__":
    #%matplotlib inline
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from matplotlib import pyplot as plt

    from tensorflow.contrib import learn
    from sklearn.metrics import mean_squared_error

    #from lstm import x_sin, sin_cos, generate_data, lstm_model

    #NOTE
    #TIMESTEPS is the track length the model is allowed to look back for.

    LOG_DIR = './ops_logs'
    TIMESTEPS = 20
    #BUG This is wrong. The number of RNN layers is not the length of data fed into the RNN
    #100 is ... ok, try fewer later
    RNN_LAYERS = [{'steps': 100}] #[{'steps': TIMESTEPS}]
    DENSE_LAYERS = None
    TRAINING_STEPS = 10000
    BATCH_SIZE = 100
    PRINT_STEPS = TRAINING_STEPS / 100

    X, y = generate_data(np.sin, np.linspace(0, 100, 10000), [(0, 1, 0, 16),
                                                            (0, 1, 0, 16),
                                                            (0, 1, 0, 16),
                                                            (0, 1, 0, 16),
                                                              ],TIMESTEPS, TIMESTEPS, seperate=False)
    #New y format breaks this
    test_sequence = np.concatenate([X['test'][0],y['test']])
    #The below is false. It still has a strange disjoint when it starts predicting though
    #BUG there is a chance the sequence generator is predicting backwards, which would explain the step at the beginning.
    #I find this strange, but there is an easy way to find out, stop feeding at a peak
    #This section here needs to be modified with a sequence generation function
    # plot_predicted, = plt.plot(predicted, label='predicted')
    # plot_test, = plt.plot(test_sequence[0:len(predicted)], label='test')
    # plt.legend(handles=[plot_predicted, plot_test])
    # plt.show()
    quit()
    X, y = generate_data(x_sin, np.linspace(0, 100, 10000), [(0,1,0,1)],TIMESTEPS, seperate=False)