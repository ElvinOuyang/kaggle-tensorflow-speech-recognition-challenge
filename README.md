# Kaggle Tensorflow Speech Recognition Challenge

This is my scripts I used for our Deep Learning final project. Our project is to finish the Kaggle Tensorflow Speech Recognition Challenge, where we need to predict the pronounced word from the recorded 1-second audio clips.

In our first research stage, we will **turn each WAV file into MFCC vector of the same dimension** (the files are of the same length). In the first few hidden layers (of either multi-layer perceptron or 1-D convolutional neural net), we plan to turn the MFCC vectors into log probability of phonemes, i.e. the basic building blocks of a pronounced word. We then plan to feed these sequences to a recurrent neural network (either a RNN or a more advanced LSTM) to train and predict the word. The assumption of this approach is that the MFCC values of a sound clip should reflect the nuance sequence in word pronunciation and the the sequence is strictly ordered. Therefore, the sequence should be be used in recurrent neural networks to classify the words.

In our second research stage, we will **turn each WAV file into a visual graph (called spectrogram) of the same size**. Since the graphical representation of the voice has pixel points of the same scale across two dimensions, we will then apply convolutional layers on the graphs to extract latent graphical patterns from the files. We will then build fully connected layers to link the extracted feature maps to the expected output. It is even possible to feed the extracted features as a sequence to a recurrent layer since the graphical patterns should also be strictly related to time series. The assumption of this approach is that the graphical pattern of different words pronounced in the WAV files should be typical enough for neural network to train on.

## STAGE 1.1: Pre-training an MLP for MFCC-phonemes layers
In our first stage, we will pre-train a model that can give a somewhat accurate probability estimation on phonemes based on MFCC values. The pre-training stage will help improve the complete model's performance since the weights of corresponding layers are not randomly initiated. Our first stab of this problem is a simple-structured MLP:

```python
MLP (
  (l1): Linear (num_inputs -> 50)
  (t1): Tanh ()
  (l2): Linear (50 -> 38)
  (t2): LogSoftmax ()
)
```

The `num_inputs` value is determined by our feature generation stage when the MFCC scores of the same length are calculated for each WAV file. The target phonemes have 38 possible labels. With a learning rate of 0.1 and a momentum of 0.9 for the Stochastic Gradient Descent optimization algorithm, we created a model with 51.75% accuracy. This performance is quite impressive for such a simple model, considering the natural probability of the classification being correct is only 1/38, i.e. 2.63%. The training logs of the first 100 epochs is displayed below:

![mlp_training_log](https://github.com/ElvinOuyang/kaggle-tensorflow-speech-recognition-challenge/blob/master/images/training_log2017_12_04_04_29_epochs_100.png)

## STAGE 1.2: Pre-training a CNN for MFCC-phonemes layers
As an alternative to the MLP training, we also built a multi-layer convolutional network with 1-D kernels for each MFCC vectors to see if we can pre-train a better model than the MLP. Since the 39 features of MFCC vectors contain MFCC scores, deltas, and delta-deltas, we restructured the input tensor to **3 by 13, where 3 is the number of channels**. By restructuring input tensors in this way, we ensure that the three types of data are updated separately in back propagation. The CNN network has a structure as below:

```python
class CNN(nn.Module):
    def __init__(self, num_inputs, num_channels):
        super(CNN, self).__init__()
        self.num_channels = num_channels
        self.feature_length = int(num_inputs / num_channels)
        self.conv1 = nn.Conv1d(num_channels, 24, 4)
        self.conv2 = nn.Conv1d(24, 48, 3)
        self.conv3 = nn.Conv1d(48, 96, 3)
        self.pool = nn.MaxPool1d(2, 2)
        self.fc1 = nn.Linear(96, 256)
        # 96 is calculated based on 13 features per channel
        self.fc2 = nn.Linear(256, 38)
        self.drop = nn.Dropout(p=0.25)
        self.t1 = nn.ReLU()
        self.t2 = nn.LogSoftmax()

    def forward(self, x):
        x = x.view(-1, self.num_channels, self.feature_length)
        x = self.t1(self.conv1(x))  # feature length 10
        x = self.t1(self.conv2(x))  # feature length 8
        x = self.drop(self.pool(x))  # feature length 4
        x = self.t1(self.conv3(x))  # feature length 2
        x = self.drop(self.pool(x))  # feature length 1, fc1 input is 1*96=96
        x = x.view(-1, 1 * 96)  # flatten feature maps to 1D
        x = self.t1(self.fc1(x))
        x = self.t2(self.fc2(x))
        return x
```

We also included pooling layers and dropout layers to ensure that the model do not overfit. The training logs of the first 50 epochs is displayed below:

![cnn_training_log](https://github.com/ElvinOuyang/kaggle-tensorflow-speech-recognition-challenge/blob/master/images/training_log_cnn_2017_12_04_18_48_epochs_50.png)

The model reached the highest accuracy at 55.55% at epoch 35.
