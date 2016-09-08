# Linear Sequence to Sequence Example in TensorFlow #

I wrote this example because I could not find any good examples of non-classification sequence to sequence (seq2seq) models written in TensorFlow. The primary example in TensorFlow focuses on text analysis. As such, most of the code is dealing with either vocabulary (embedding) or varying sentence lengths (bucketing) which can obfiscate the underlying tutorial with regards to sequencing. To simplify the structure of the model in this tutorial, a single value prediction model is used. 

Sequence generation is often used in language processing, but has other uses as well, such as handwriting generation [(Alex Graves)](https://arxiv.org/abs/1308.0850), and very recently, pedestrian path prediction [(Alahi et. al.)](http://web.stanford.edu/~alahi/downloads/CVPR16_N_LSTM.pdf). Both of these implementations use [Mixture Density Networks](http://web.stanford.edu/~alahi/downloads/CVPR16_N_LSTM.pdf) as the output layer, which is outside the scope of this tutorial.

The example problem here will be to predict a repeating function, such as a sine or square wave. This is a time domain problem. As such, there is a section of the sequence that exists in the past, and a section in the future, for any given time step.

## Sequence to Sequence Recurrent Neural Networks ##

The primary difference between a standard model and a seq2seq model is the recurrent layer. A recurrent layer is a layer that takes input from both the data being fed into the model, and data the model passes to itself between timesteps (interal states).

There are many different types of Recurrent Neural Networks (RNN), such as one to many, many to one, and many to many. [Karpathy](http://web.stanford.edu/~alahi/downloads/CVPR16_N_LSTM.pdf) provides a great explanation. In the TF tool library there is a discrete handover point between observing the input (encoder) data, and generating output (decoder) data. This point is marked in the model by the "GO" symbol.


![alt text][Basic RNN]
[Basic RNN]: basic_rnn.png

At the centre of this model is an RNN.  This consists of encoder inputs (blue), decoder inputs (green), and decoder outputs (red). The advantage of an RNN is that the encoder sequence and decoder sequence do not have to be of fixed length. For this model, the lengths have to be declared during creation. The model can be saved, and re-created with different sequence lengths, to continue either training or testing.

![alt text][Seq2Seq Linear Model]
[Seq2Seq Linear Model]: seq2seq_lin_model.png

This is the overall architecture used in this example. Remember the goal here is to observe a time sequence from a waveform, and predict the future waveform. Here, the past sequence is fed into the model, possibly with noise added. Once the sequence is fully passed into the model, a "GO" token is passed into the model, to mark the beginning of the prediction. At each time step of the model, the output is taken, passed through the output projection layer, and then fed back into the model as the input for the next timestep. This loopback method continues until the prediciton sequence is of desired length. Finally, the predicted sequence is compared to the ground truth future sequence, to generate the loss.

The output projection layer is a linear layer used to transform the output of the RNN into the final output of the model. In this case, we are predicting a single number, so the output projection layer must reduce the complexity of the output down to this size. Without this layer, the model output would be as large as the number of nodes in the layers of the RNN. 

The "GO" token here is just a large negative number, larger than any input value. In language processing, it would be another word in the vocabulary (just like how spacebar is another letter on a keyboard). While this token is useful to mark when a RNN should start writing a sentence, I do not believe it is appropriate in a time based sequence, as it allows a RNN step to exist that does not correspond to a real timestep. However, I could not find a clean way to remove it given the current Seq2Seq tools.  

### Data Generation ###
The data generated for testing this model is of a repeating function, here a square wave is used. A sample taken from the sequence consists of the past sequence, and the future sequence. There is a flag in the code to generate square waves with random properties, or to only sample from a single waveform. The idea behind this is to test if it can reproduce a square wave of any frequency/amplitude, or only a specific fequency/amplitude.

![alt text][Example Wave]
[Example Wave]: example_wave.png

### Loss Function ###
As a whole predictive track is generated by the model, the whole sequence is compared to the true data to generate a loss. This is different from other sequence generation techniques, such as [(Alex Graves)](https://arxiv.org/abs/1308.0850), where only the next timestep is compared. In this model, a pairwise comparison is used, and is reduced by a root mean square error to produce a single value.

## TensorFlow Library Modification 

The `basic_rnn_seq2seq()` function does not support a loop function. The function makes a call to `rnn_decoder()`, which does allow for a loop function. So I've made this change, and included it in `TF_mods.py` as `basic_rnn_seq2seq_with_loop_function()`.

```
def basic_rnn_seq2seq_with_loop_function(
        encoder_inputs, decoder_inputs, cell, dtype=dtypes.float32, loop_function=None, scope=None):
    """Basic RNN sequence-to-sequence model. Edited for a loopback function. """
    with variable_scope.variable_scope(scope or "basic_rnn_seq2seq_with_loop_function"):
        _, enc_state = rnn.rnn(cell, encoder_inputs, dtype=dtype)
        return rnn_decoder(decoder_inputs, enc_state, cell, loop_function=loop_function)
```

# Results

The system can sucessfully learn on both sine and square waves of fixed frequency / amplitude. Producing a square wave proves that the system is able to 'count' a number of timesteps, and then flip states. However, when randomising the parameters of the input waveform the system will commonly be stable for the length of the training decoder steps, and then collapse into a smaller value. Further investigation required. Perhaps varying the training lengths (using bucketing?) may prove useful.

## Successful Square Wave
![alt text][Square Wave]
[Square Wave]: S2Ssqrtre30-trd40-tse100-tsd500-rnn32-nl3-bs64-lr0.5-ld0.7-randF.png

## Successful Sine Wave
![alt text][Sine Wave]
[Sine Wave]: S2Ssintre30-trd40-tse100-tsd500-rnn32-nl3-bs32-lr0.5-ld0.7-randF.png

## Failed Sine Wave, Randomized Wave Params for Training Set
![alt text][Failed Random Sine Wave]
[Failed Random Sine Wave]: S2Ssintre30-trd40-tse100-tsd500-rnn64-nl1-bs16-lr0.5-ld0.7-randT.png

# Usage

Run train.py. The system will train on a given square wave, and then generate an output graph.

# Copyright / License
Copyright: Alex Zyner (2016), License WTFPL, http://www.wtfpl.net/