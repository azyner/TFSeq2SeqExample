
from seq2seq_model import Seq2SeqModel
import tensorflow as tf
import numpy as np
import data_utils

import math
import os
import random
import sys
import time


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.7,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 32,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("rnn_size", 16, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_string("data_dir", "data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory.")
tf.app.flags.DEFINE_string("logs_dir", "logs", "Logs directory.")
tf.app.flags.DEFINE_string("plot_dir", "plot", "Output plots directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("predict", False,
                            "Set to True to use the model to generate a sequence prediction.")
tf.app.flags.DEFINE_boolean("run_many", True,
                            "Run a list of many jobs")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")
tf.app.flags.DEFINE_boolean("gen_random_input_data", False,
                            "Generate data from function using varying random parameters (True) or a constant, single function")
tf.app.flags.DEFINE_integer("train_observation_steps", 30, "How many steps of data to feed the model during training.")
tf.app.flags.DEFINE_integer("train_prediction_steps", 40, "How many steps of data the model generates during training.")
tf.app.flags.DEFINE_integer("test_observation_steps", 100, "How many steps of data the model generates during testing.")
tf.app.flags.DEFINE_integer("test_prediction_steps", 500, "How many steps of data the model generates during testing.")

FLAGS = tf.app.flags.FLAGS

def get_title_from_params():
    return ('S2S' +
            'tre' + str(FLAGS.train_observation_steps) + '-'
            'trd' + str(FLAGS.train_prediction_steps) + '-'
            'tse' + str(FLAGS.test_observation_steps) + '-'
            'tsd' + str(FLAGS.test_prediction_steps) + '-'
            'rnn' + str(FLAGS.rnn_size) + '-'
            'nl'  + str(FLAGS.num_layers) + '-'
            'bs' + str(FLAGS.batch_size) + '-'
            'lr' + str(FLAGS.learning_rate)+ '-'
            'ld' + str(FLAGS.learning_rate_decay_factor) +'-'
            'rand' + ('T' if FLAGS.gen_random_input_data else 'F'))

def gen_data(observation_steps, prediction_steps):
    random.seed = 42
    num_functions = 20 #number of different functions
    function_set = []
    if FLAGS.gen_random_input_data:
        #function tuple is in order: a+b*fct(c+d*x)
        for i in range(num_functions):
            function_set.append((
                random.choice(np.linspace(-0.5,0.5,10)), #amplitude offset
                random.choice(np.linspace(0.1,1.5,10)), #amplitude
                random.choice(np.linspace(0,10,10)), #frequency offset
                random.choice(np.linspace(8,32,24)),# frequency (/2pi)
            ))
    function_set.append((0, 1, 0, 16))

    import scipy.signal as spsig

    #fct = np.sin
    #fct = spsig.sawtooth
    fct = spsig.square

    def doublesin(t):
        return 0.9 * np.sin(t) + 0.1*np.sin(10*t)

    def doublesquare(t):
        return 0.5*spsig.square(t,duty=0.75) - 0.5*spsig.square(2*t,duty=0.25)

    return data_utils.generate_data(fct, np.linspace(0, 100, 10000), function_set,
                                    observation_steps, prediction_steps, seperate=False)

def create_model(session, feed_forward, train_model, observation_steps, prediction_steps, batch_size, 
                 rnn_size, num_layers, learning_rate, learning_rate_decay_factor, input_size, max_gradient_norm):
    parameters = None
    model = Seq2SeqModel(parameters, feed_forward, train_model, observation_steps, prediction_steps, batch_size,
                         rnn_size, num_layers, learning_rate, learning_rate_decay_factor, input_size, max_gradient_norm)
    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)
    if not os.path.exists(os.path.join(FLAGS.train_dir,get_title_from_params())):
        os.makedirs(os.path.join(FLAGS.train_dir,get_title_from_params()))
    ckpt = tf.train.get_checkpoint_state(os.path.join(FLAGS.train_dir,get_title_from_params()))
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model

def train():
    tf.reset_default_graph()
    print "Start training for: " + get_title_from_params()
    """Train a en->fr translation model using WMT data."""
    # Prepare WMT data.
    #PREPARE DATA INTO:
    #[train test validate] x [input_sequences, output_sequences]

    with tf.Session() as sess:
        # Create model.
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.rnn_size))

        #allowed to be varied between training and decoding
        past_steps = FLAGS.train_observation_steps
        future_steps = FLAGS.train_prediction_steps

        #Set for training
        train_model = True
        feed_forward = False
        input_size = 1 #TODO fix this with sizing the input

        model = create_model(sess, feed_forward, train_model, past_steps, future_steps, FLAGS.batch_size,
                             FLAGS.rnn_size, FLAGS.num_layers,FLAGS.learning_rate,FLAGS.learning_rate_decay_factor, input_size, FLAGS.max_gradient_norm)

        train_writer=tf.train.SummaryWriter(os.path.join(FLAGS.logs_dir,'train'+get_title_from_params()),sess.graph)
        summary_op = tf.merge_all_summaries()

        # Read data into buckets and compute their sizes.
        print ("Reading development and training data (limit: %d)."
               % FLAGS.max_train_data_size)

        past_sequence_data, future_sequence_data = gen_data(past_steps, future_steps)

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            # Get a batch and make a step.
            start_time = time.time()

            past_sequences, future_sequences, target_weights = model.get_batch(past_sequence_data['train'],future_sequence_data['train'])

            observations = past_sequences #Apply noise here if desired

            _, step_loss, _ = model.step(sess, observations, future_sequences,
                                         target_weights, """bucket_id""", feed_forward, train_model)
            #Periodically, run without training for the summary logs
            if current_step % 20 == 0:
                _, step_loss, _ = model.step(sess, observations, future_sequences,
                                         target_weights, """bucket_id""", feed_forward, False,summary_writer=train_writer)

            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                
                # Print statistics for the previous epoch.
                perplexity = (loss) if loss < 300 else float("inf")
                print ("global step %d learning rate %.4f step-time %.2f Batch average MSE loss "
                       "%.4f" % (model.global_step.eval(), model.learning_rate.eval(),
                                 step_time, perplexity))
                
                # Decrease learning rate if no improvement was seen over last 3 times.
                decrement_timestep = 3
                if len(previous_losses) > decrement_timestep-1 and loss > 0.95*(max(previous_losses[-decrement_timestep:])): #0.95 is float fudge factor
                  sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(os.path.join(FLAGS.train_dir,get_title_from_params()), "TFseq2seqSinusoid.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                if perplexity < 0.02 or model.learning_rate.eval() < 0.01:
                    break


def predict():

    tf.reset_default_graph()

    with tf.Session() as sess:
        # Create model and load parameters.
        #can be varied between training and decoding
        past_steps = FLAGS.test_observation_steps
        future_steps = FLAGS.test_prediction_steps

        #set for decoding
        train_model = False
        feed_forward = False
        input_size = 1

        model = create_model(sess, feed_forward, train_model, past_steps, future_steps, 1,
                             FLAGS.rnn_size, FLAGS.num_layers, FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
                             input_size, FLAGS.max_gradient_norm)
        model.batch_size = 1  # One sequence for testing

        past_sequence_data, future_sequence_data = gen_data(past_steps, future_steps)
        past_sequences, future_sequences, target_weights = model.get_batch(past_sequence_data['test'], future_sequence_data['test'])

        observations = past_sequences  # Apply noise here if desired

        true_output = np.copy(future_sequences)

        # Force all decoder inputs after the 'go' symbol to zero. They should be ignored, but this is just to be sure
        for i in range(future_steps):
            future_sequences[i+1][0] = 0

        _, output_loss, output_prediction = model.step(sess,observations,future_sequences,target_weights,'''bucket_id''',
                                                          feed_forward, train_model)

        #re-format graph input
        input_plot = []
        for l in range(len(observations)):
            input_plot.append(observations[l][0])
        output_gen_plt = []
        for l in range(len(output_prediction)):
            output_gen_plt.append(np.average(output_prediction[l][0]))
        #Here we discard the GO symbol
        true_output_plot = []
        for l in range(len(true_output)-1):
            true_output_plot.append(true_output[l+1][0])

        # Get plot ranges
        y_range = np.linspace(data_utils.data_linspace_tuple[0],
                              data_utils.data_linspace_tuple[1],
                              data_utils.data_linspace_tuple[2])
        input_range = y_range[0:len(input_plot)]
        output_range = y_range[len(input_plot):len(input_plot)+len(output_prediction)]
        plt_title = "TFSeq2Seq" + "rnn_size " + str(FLAGS.rnn_size) + " n_layers " + str(FLAGS.num_layers)

        if False: #Plot HTML bokeh
            from bokeh.plotting import figure, output_file, show
            output_file("traces.html")
            p1 = figure(title=plt_title, x_axis_label='x', y_axis_label='y',
                        plot_width=800, plot_height=800)  # ~half a 1080p screen
            p1.line(input_range, input_plot, legend="Input.", line_width=2,color='black')
            p1.line(output_range, true_output_plot, legend="True Output.", line_width=2,color='blue')
            p1.line(output_range, output_gen_plt, legend="Generated Output.", line_width=2,color='red')
            show(p1)

        if True: #Use matplotlib to plot PNG
            if not os.path.exists(FLAGS.plot_dir):
                os.makedirs(FLAGS.plot_dir)
            legend_str = []
            import matplotlib.pyplot as plt
            plt.figure(figsize=(20,10))
            plt.plot(input_range, input_plot)
            legend_str.append(['Input'])
            plt.plot(output_range, true_output_plot)
            legend_str.append(['True Output'])
            plt.plot(output_range, output_gen_plt)
            legend_str.append(['Generated Output'])
            plt.legend(legend_str, loc='upper left')
            fig_path = os.path.join(FLAGS.plot_dir,get_title_from_params()+'.png')
            plt.savefig(fig_path,bbox_inches='tight')
            #plt.show()

def run_many():

    rnn_size_range = [2,4,8,16,32,64]
    num_layers_range = [1,2,3]
    random_range = [True, False]
    batch_size_range = [16, 32, 64]
    learning_rate_range = [0.5]

    for batch_size in batch_size_range:
        for learning_rate in learning_rate_range:
            for size in rnn_size_range:
                for layers in num_layers_range:
                    for random in random_range:
                        FLAGS.rnn_size = size
                        FLAGS.num_layers = layers
                        FLAGS.gen_random_input_data = random
                        FLAGS.batch_size = batch_size
                        FLAGS.learning_rate = learning_rate
                        train()
                        predict()

def main(_):
    if FLAGS.run_many:
        run_many()
    elif FLAGS.predict:
        predict()
    else:
        train()
        predict()

if __name__ == "__main__":
    tf.app.run()
