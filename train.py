#THIS IS A HEAVILY MODIFIED VERSION OF:
#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/translate.py

#load params

#sort data into feed_dicts

#training loop, report losses, etc

#declare session, checkpoints, loading etc
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
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.9,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("en_vocab_size", 40000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("fr_vocab_size", 40000, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory.")
tf.app.flags.DEFINE_string("logs_dir", "logs", "Logs directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS



def create_model(session,feed_forward, train_model, encoder_steps, decoder_steps, batch_size):
    parameters = None
    model = Seq2SeqModel(parameters,feed_forward, train_model, encoder_steps, decoder_steps, batch_size)
    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model



def train():
  """Train a en->fr translation model using WMT data."""
  # Prepare WMT data.
  #PREPARE DATA INTO:
  #[train test validate] x [input_tracks, output_tracks]

  with tf.Session() as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))

    encoder_steps = 40
    decoder_steps = 50
    batch_size = 64
    train_model = True
    feed_forward = False

    model = create_model(sess, feed_forward, train_model, encoder_steps, decoder_steps, batch_size)

    train_writer=tf.train.SummaryWriter(os.path.join(FLAGS.logs_dir,'train'),sess.graph)
    summary_op = tf.merge_all_summaries()

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)
    #dev_set = test_data?
    #train_set = get_data

    X, y = data_utils.generate_data(np.sin, np.linspace(0, 100, 10000), [(0, 1, 0, 16),
                                                              (0, 1, 0, 16),
                                                              (0, 1, 0, 16),
                                                              (0, 1, 0, 16),
                                                              ], encoder_steps, decoder_steps, seperate=False)
    #X['test'][9960][23]
    #y['test'][9960][31]
    #Replace with minibatch sizes?

    #train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    #train_total_size = float(sum(train_bucket_sizes))

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    while True:

      # Get a batch and make a step.
      start_time = time.time()

      encoder_inputs, decoder_inputs, target_weights = model.get_batch(X['train'],y['train'])

      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, """bucket_id""", feed_forward, train_model)
      #Periodically, run without training for the summary logs
      if current_step % 20 == 0:
          _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, """bucket_id""", feed_forward, False,summary_writer=train_writer)

      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        perplexity = (loss) if loss < 300 else float("inf")
        print ("global step %d learning rate %.4f step-time %.2f Batch average MSE loss "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
        # Decrease learning rate if no improvement was seen over last 3 times.
        decrement_timestep = 3
        if len(previous_losses) > decrement_timestep-1 and loss > max(previous_losses[-decrement_timestep:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, "TFseq2seqSinusoid.ckpt")


        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0


        # Run evals on development set and print their perplexity.

        # encoder_inputs, decoder_inputs, target_weights = model.get_batch(
        #   dev_set, bucket_id)
        # _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
        #                            target_weights, bucket_id, True)
        # eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
        #   "inf")
        # print("  eval: run %d perplexity %.2f" % (iteration???, eval_ppx))
        # sys.stdout.flush()


def decode():
  with tf.Session() as sess:
    # Create model and load parameters.
    encoder_steps = 40
    decoder_steps = 50

    train_model = False
    feed_forward = False
    # HACK

    model = create_model(sess, feed_forward, train_model, encoder_steps, decoder_steps, 1)
    model.batch_size = 1  # One string for testing

    X, y = data_utils.generate_data(np.sin, np.linspace(0, 100, 10000), [(0, 1, 0, 16),
                                                                   (0, 1, 0, 16),
                                                                   (0, 1, 0, 16),
                                                                   (0, 1, 0, 16),
                                                                   ], encoder_steps, decoder_steps, seperate=False)
    encoder_inputs, decoder_inputs, target_weights = model.get_batch(X['train'], y['train'])
    true_output = np.copy(decoder_inputs)
    for i in range(decoder_steps):
        decoder_inputs[i+1][0] = 0
    output_a, output_loss, output_logits = model.step(sess,encoder_inputs,decoder_inputs,target_weights,'''bucket_id''',
                                                      feed_forward, train_model)

    print output_logits
      #I see two issues here:
      #1: training with feed forward is bad
      #2: the output is a list that is rnn_size wide. I need output projection?

      #Fix in this order
      #I created a monster. The average of all input_size is the correct answer
    output = []
    for l in range(len(output_logits)):
        output.append(np.average(output_logits[l]))
    print output
    print true_output - output
    print 'start'
    for l in range(len(output)):
        print true_output[l][0][0],  ', ',  output[l]
    print'stop'

    #re-format graph input
    input_plot = []
    for l in range(len(encoder_inputs)):
        input_plot.append(encoder_inputs[l])
    output_gen_plt = []
    for l in range(len(output_logits)):
        output_gen_plt.append(np.average(output_logits[l][0]))
    true_output_plot = []
    for l in range(len(true_output)):
        true_output_plot.append(true_output[l])

    input_range = range(0,len(input_plot))
    output_range = range(len(input_plot),len(input_plot)+len(output_logits))

    from bokeh.plotting import figure, output_file, show
    output_file("traces.html")
    p1 = figure(title="TFSeq2Seq", x_axis_label='x', y_axis_label='y')
    p1.line(input_range, input_plot, legend="Input.", line_width=2,color='black')
    p1.line(output_range, true_output_plot, legend="True Output.", line_width=2,color='blue')
    p1.line(output_range, output_gen_plt, legend="Generated Output.", line_width=2,color='red')
    show(p1)
    print 'break'


def self_test():
  """Test the translation model."""
  with tf.Session() as sess:
    print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                       5.0, 32, 0.3, 0.99, num_samples=8)
    sess.run(tf.initialize_all_variables())

    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
    for _ in xrange(5):  # Train the fake model for 5 steps.
      bucket_id = random.choice([0, 1])
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          data_set, bucket_id)
      model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                 bucket_id, False)



def main(_):
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()