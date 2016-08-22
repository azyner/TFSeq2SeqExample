
# Special vocabulary symbols. Artifact from the vocab system. I don't know a good way to replace this in a linear system

_PAD = 0.0
_GO = 999
#_EOS = b"_EOS"
#_UNK = b"_UNK"
#_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


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


def rnn_data(data, time_steps, labels=False):
    """
    creates new data frame based on previous observation
      * example:
        l = [1, 2, 3, 4, 5]
        time_steps = 2
        -> labels == False [[1, 2], [2, 3], [3, 4]]
        -> labels == True [2, 3, 4, 5]
    """
    rnn_df = []
    for i in range(len(data) - time_steps):
        if labels:
            try:
                rnn_df.append([data.iloc[i + time_steps].as_matrix()])
            except AttributeError:
                rnn_df.append(data.iloc[i + time_steps])
        else:
            data_ = data.iloc[i: i + time_steps].as_matrix()
            rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])
    return np.array(rnn_df)


def split_data(data, val_size=0.1, test_size=0.1):
    """
    splits data to training, validation and testing parts
    """
    ntest = int(round(len(data) * (1 - test_size)))
    nval = int(round(len(data.iloc[:ntest]) * (1 - val_size)))

    df_train, df_val, df_test = data.iloc[:nval], data.iloc[nval:ntest], data.iloc[ntest:]

    return df_train, df_val, df_test


def prepare_data(data, time_steps, labels=False, val_size=0.1, test_size=0.1):
    """
    Given the number of `time_steps` and some data,
    prepares training, validation and test data for an lstm cell.
    """
    df_train, df_val, df_test = split_data(data, val_size, test_size)
    return (rnn_data(df_train, time_steps, labels=labels),
            rnn_data(df_val, time_steps, labels=labels),
            rnn_data(df_test, time_steps, labels=labels))

def generate_data(fct, x, fct_mod, time_steps, seperate=False):
    """generates data with based on a function fct"""
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
            w_train_x, w_val_x, w_test_x = prepare_data(data['a'] if seperate else data, time_steps)
            w_train_y, w_val_y, w_test_y = prepare_data(data['b'] if seperate else data, time_steps, labels=True)
            train_x.extend(w_train_x)
            val_x.extend(w_val_x)
            test_x.extend(w_test_x)
            train_y.extend(w_train_y)
            val_y.extend(w_val_y)
            test_y.extend(w_test_y)
        else:
            #training / val are most of data. Test is the last function.
            if(wave is not fct_mod[-1]):
                if not isinstance(data, pd.DataFrame):
                    data = pd.DataFrame(data)
                w_train_x, w_val_x, w_test_x = prepare_data(data['a'] if seperate else data, time_steps,test_size = 0)
                w_train_y, w_val_y, w_test_y = prepare_data(data['b'] if seperate else data, time_steps,test_size = 0, labels=True)
                train_x.extend(w_train_x)
                val_x.extend(w_val_x)
                train_y.extend(w_train_y)
                val_y.extend(w_val_y)
            else:
                if not isinstance(data, pd.DataFrame):
                    data = pd.DataFrame(data)
                test_x.extend(rnn_data(data, time_steps, labels=False))
                test_y.extend(rnn_data(data, time_steps, labels=True))

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

#Everything in learn is actually skflow. This is why this is so confusing, and why the session is abstracted.

regressor = learn.TensorFlowEstimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS), n_classes=0,
                                      verbose=2,  steps=TRAINING_STEPS, optimizer='Adagrad',
                                      learning_rate=0.03, batch_size=BATCH_SIZE)

X, y = generate_data(np.sin, np.linspace(0, 100, 10000), [#(0,  1, 0, 1),
                                                          #(0,  2, 0, 5),
                                                          #(0,  1, 1, 0.5),
                                                          #(0,  1, 0, 5),
                                                          #(1,  1, 0, 10),
                                                          #(1,  3, 0, 5),
                                                          #(1,  1, 1, 0.1),
                                                          #(1,  2, 0, 5),
                                                          #(1,  1, 0, 50),
                                                          #(-1, 2, 0, 5),
                                                          #(-1, 1, 1, 20),
                                                          #(-1, 1, 0, 5),
                                                          #(-1, 0.1, 0, 10),
                                                          #(-1, 0.5, 0, 12),
                                                          #(0,  -1, 0, 1),
                                                          #(0,  -2, 0, 5),
                                                          #(0,  -1, 1, 0.5),
                                                          #(0,  -1, 0, 5),
                                                          #(1,  -1, 0, 10),
                                                          #(1,  -3, 0, 5),
                                                          #(1,  -1, 1, 0.1),
                                                          #(1,  -2, 0, 5),
                                                          #(1,  -1, 0, 50),
                                                          #(-1, -2, 0, 5),
                                                          #(-1, -1, 1, 20),
                                                          #(-1, -1, 0, 5),
                                                          #(-1, -0.1, 0, 10),
                                                          #(-1, 1.5, 0, 16),
                                                        (0, 1, 0, 16),
                                                        (0, 1, 0, 16),
                                                        (0, 1, 0, 16),
                                                        (0, 1, 0, 16),
                                                          ],TIMESTEPS, seperate=False)
#New y format breaks this
test_sequence = np.concatenate([X['test'][0],y['test']])

# create a lstm instance and validation monitor #Val monitor has data in an incorrect format?
validation_monitor = learn.monitors.ValidationMonitor(X['val'], y['val'],
                                                      every_n_steps=PRINT_STEPS,
                                                      early_stopping_rounds=1000)#early termination disable
regressor.fit(X['train'], y['train'], monitors=[validation_monitor])#, logdir=LOG_DIR)


predicted = regressor.predict(X['test'])
predicted = generate_sequence(regressor,test_sequence,100,50)

#The below is false. It still has a strange disjoint when it starts predicting though
#BUG there is a chance the sequence generator is predicting backwards, which would explain the step at the beginning.
#I find this strange, but there is an easy way to find out, stop feeding at a peak
#This section here needs to be modified with a sequence generation function

rmse = np.sqrt(((predicted - test_sequence[0:len(predicted)]) ** 2).mean(axis=0))
score = mean_squared_error(predicted, test_sequence[0:len(predicted)])
print ("MSE: %f" % score)

plot_predicted, = plt.plot(predicted, label='predicted')
plot_test, = plt.plot(test_sequence[0:len(predicted)], label='test')
plt.legend(handles=[plot_predicted, plot_test])
plt.show()

quit()



X, y = generate_data(x_sin, np.linspace(0, 100, 10000), [(0,1,0,1)],TIMESTEPS, seperate=False)

