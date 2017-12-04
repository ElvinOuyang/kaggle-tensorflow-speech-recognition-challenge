# Kaggle Tensorflow Speech Recognition Chanllege

This is my scripts I used for our Deep Learning final project. Our project is to finish the Kaggle Tensorflow Speech Recognition Challenge, where we need to predict the pronounced word from the recorded 1-second audio clips.

In our first research stage, we will **turn each WAV file into MFCC vector of the same dimension** (the files are of the same length). In the first few hidden layers (of either multi-layer perceptron or 1-D convolutional neural net), we plan to turn the MFCC vectors into log probability of phonemes, i.e. the basic building blocks of a pronounced word. We then plan to feed these sequences to a recurrent neural network (either a RNN or a more advanced LSTM) to train and predict the word. The assumption of this approach is that the MFCC values of a sound clip should reflect the nuance sequence in word pronunciation and the the sequence is strictly ordered. Therefore, the sequence should be be used in recurrent neural networks to classify the words.

In our second research stage, we will **turn each WAV file into a visual graph (called spectrogram) of the same size**. Since the graphical representation of the voice has pixel points of the same scale across two dimensions, we will then apply convolutional layers on the graphs to extract latent graphical patterns from the files. We will then build fully connected layers to link the extracted feature maps to the expected output. It is even possible to feed the extracted features as a sequence to a recurrent layer since the graphical patterns should also be strictly related to time series. The assumption of this approach is that the graphical pattern of different words pronounced in the WAV files should be typical enough for neural network to train on.

## STAGE 1: Pre-training an MLP for MFCC-phonemes layers
In our first stage, we will pre-train a model that can give a somewhat accurate probability estimation on phonemes based on MFCC values. The pre-training stage will help improve the complete model's performance since the weights of corresponding layers are not randomly initiated. Our first stab of this problem is a simple-structured MLP:

```python
MLP (
  (l1): Linear (num_inputs -> 50)
  (t1): Tanh ()
  (l2): Linear (50 -> 38)
  (t2): LogSoftmax ()
)
```

The `num_inputs` value is determined by our feature generation stage when the MFCC scores of the same length are calculated for each WAV file. The target phonemes have 38 possible labels. With a learning rate of 0.1 and a momentum of 0.9 for the Stochastic Gradient Descent optimization algorithm, we created a model with around 50% accuracy and 1.5 average NLLLoss. This performance is quite impressive for such a simple model, considering the natural probability of the classification being correct is only 1/38, i.e. 2.63%. The training logs of the first 100 epochs is displayed below:

![mlp_training_log](https://github.com/ElvinOuyang/kaggle-tensorflow-speech-recognition-challenge/blob/master/images/training_log2017_12_04_04_29_epochs_100.png)
