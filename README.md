# Kaggle Tensorflow Speech Recognition Challenge

This is my scripts I used for our Deep Learning final project. Our project is to finish the Kaggle Tensorflow Speech Recognition Challenge, where we need to predict the pronounced word from the recorded 1-second audio clips.

In our first research stage, we will **turn each WAV file into MFCC vector of the same dimension** (the files are of the same length). In the first few hidden layers (of either multi-layer perceptron or 1-D convolutional neural net), we plan to turn the MFCC vectors into log probability of phonemes, i.e. the basic building blocks of a pronounced word. We then plan to feed these sequences to a recurrent neural network (either a RNN or a more advanced LSTM) to train and predict the word. The assumption of this approach is that the MFCC values of a sound clip should reflect the nuance sequence in word pronunciation and the the sequence is strictly ordered. Therefore, the sequence should be be used in recurrent neural networks to classify the words.

In our second research stage, we will **turn each WAV file into a visual graph (called spectrogram) of the same size**. Since the graphical representation of the voice has pixel points of the same scale on two dimensions, we will then apply convolutional layers on the graphs to extract latent graphical patterns from the files. We will then build fully connected layers to link the extracted feature maps to the expected output. It is even possible to feed the extracted features as a sequence to a recurrent layer since the graphical patterns should also be strictly related to time series, but the model might be too complicated for the simple task we are dealing with. The assumption of this approach is that the graphical patterns in the spectrograms of different words pronounced in the WAV files should be typical enough for the convolutional neural network to train on.

In the third and final stage, we will complete the Kaggle challenge by bringing in "silent" audio clips that only includes 1 second background noise. By adding the extra layer of information about random background noise, we expect the model to better distinguish the graphical representation between useful voice and background noise. Additionally, since the real world application of the model is to distinguish the voice commands from other "unknown" words, we will also need to adjust the sampling strategy to have the neural network learn more of the "positive" outcome.

## STAGE 1: Proof-of-concept training for MFCC-phonemes approach

### STAGE 1.1: Pre-training an MLP for MFCC-phonemes layers

In our first stage, I will pre-train a model that can give a somewhat accurate probability estimation on phonemes based on MFCC values. The pre-training stage will help improve the complete model's performance since the weights of corresponding layers are not randomly initiated. My first stab of this problem is a simple-structured MLP:

```python
MLP (
  (l1): Linear (num_inputs -> 50)
  (t1): Tanh ()
  (l2): Linear (50 -> 38)
  (t2): LogSoftmax ()
)
```

The `num_inputs` value is determined by our feature generation stage when the MFCC scores of the same length are calculated for each WAV file. The target phonemes have 38 possible labels. With a learning rate of 0.1 and a momentum of 0.9 for the Stochastic Gradient Descent optimization algorithm, I created a model with 51.75% accuracy. This performance is quite impressive for such a simple model, considering the natural probability of the classification being correct is only 1/38, i.e. 2.63%. The training logs of the first 100 epochs is displayed below:

![mlp_training_log](https://github.com/ElvinOuyang/kaggle-tensorflow-speech-recognition-challenge/blob/master/images/training_log2017_12_04_04_29_epochs_100.png)

### STAGE 1.2: Pre-training a CNN for MFCC-phonemes layers

As an alternative to the MLP training, I also built a multi-layer convolutional neural network with 1-D kernels for each MFCC vector inputs to see if I can pre-train a better model than the MLP. Since the 39 features of MFCC vectors contain MFCC scores, deltas, and delta-deltas, I restructured the input tensor to **3 by 13, where 3 is the number of channels**. By restructuring input tensors in this way, I can ensure that the three types of data are updated separately in the back propagation process. The CNN network has a structure as below:

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

I also included pooling layers and dropout layers to ensure that the model do not overfit. The training logs of the first 50 epochs is displayed below:

![cnn_training_log](https://github.com/ElvinOuyang/kaggle-tensorflow-speech-recognition-challenge/blob/master/images/training_log_cnn_2017_12_04_18_48_epochs_50.png)

The model reached the highest accuracy at 55.55% at epoch 35.

### STAGE 1.3: Concluding the MFCC-phonemes approach

The suboptimal result from the primary testing with MFCC and phonemes indicates that this traditional method is not the ideal approach that we should follow using neural network. I then seek to generate an easier network with our second approach.

## STAGE 2: Training a CNN for spectrogram-word predictions

### STAGE 2.1: Generating spectrograms from .WAV files

In order to train a convolutional network on graphical representation of .WAV files, I will need to preprocess the files. I used `scipy` package to generate the spectrograms in PNG format and generated a map file that includes the path of the PNG files and their corresponding labels. The core functions I used in this stage are as below:

```python
def log_specgram(audio, sample_rate, window_size=20, step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, _, spec = signal.spectrogram(audio,
                                        fs=sample_rate,
                                        window='hann',
                                        nperseg=nperseg,
                                        noverlap=noverlap,
                                        detrend=False)
    return freqs, np.log(spec.T.astype(np.float32) + eps)

def wav2img(wav_path, targetdir='', figsize=(4, 4)):
    """
    takes in wave file path
    and the fig size. Default 4,4 will make images 288 x 288
    """
    fig = plt.figure(figsize=figsize)
    samplerate, test_sound = wavfile.read(wav_path)
    _, spectrogram = log_specgram(test_sound, samplerate)
    output_file = wav_path.split('/')[-1].split('.wav')[0]
    output_file = targetdir + '/' + output_file
    plt.imsave('%s.png' % output_file, spectrogram)
    plt.close()
    return (output_file + '.png')
```

The generated spectrograms are as below:

![spectrogram_samples](https://github.com/ElvinOuyang/kaggle-tensorflow-speech-recognition-challenge/blob/master/images/spectrogram_samples.png)

From these samples, I have seen that different words' spectrograms do look differently. I am hopeful that we can train a convolutional network on these inputs! The other key part of my next training stage is the map file. The generated map file has contents as below:

```shell
                                              path target
0  ../data/train/audio/right/988e2f9a_nohash_0.wav  right
1  ../data/train/audio/right/1eddce1d_nohash_3.wav  right
2  ../data/train/audio/right/93ec8b84_nohash_0.wav  right
3  ../data/train/audio/right/6272b231_nohash_1.wav  right
4  ../data/train/audio/right/439c84f4_nohash_1.wav  right
```

I will use the path and the target from this map file in my PyTorch stochastic mini batch training paradigm in the following stages.

### STAGE 2.2: Building initial CNN for spectrogram pictures and labels

In my training script, I customized `pytorch` package's handy `DataSet` class and `DataLoader` class to achieve multi-CPU fast loading and easy GPU integration. The main script for the CNN training consists of the following steps:

* Define hyper parameters for epochs, GPU usage, CPU usage, and optimization
* Load map file and define label index to replace labels with numbers
* Conduct train test split on map file
* Define spectrogram dataset and spectrogram PNG file transform methods so that PNG files and targets are automatically loaded and preprocessed into tensors of the same shape (batch_size * 3 * 99 * 161 and batch_size * 1)
* Create training and validating dataset and dataloader, where mini-batches are automatically generated using multiple CPUs
* Define CNN network structure, loss/criterion function, and optimization algorithm
* Define training, validating, and testing functions
* Write epoch loops with training, validation, and log recording

The core part of the script, i.e. the CNN network framework, is defined as below:

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, 4)
        self.conv2 = nn.Conv2d(24, 48, 3)
        self.conv3 = nn.Conv2d(48, 96, 4)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(10 * 18 * 96, 256)
        self.fc2 = nn.Linear(256, 11)
        self.drop = nn.Dropout(p=0.25)
        self.pad = nn.ZeroPad2d((0, 1, 0, 0))
        self.t1 = nn.ReLU()
        self.t2 = nn.LogSoftmax()

    def forward(self, x):
        x = self.t1(self.conv1(x))  # feature shape 96 * 158
        x = self.drop(self.pool(x))  # feature shape 48 * 79
        x = self.t1(self.conv2(x))  # feature shape 46 * 77
        x = self.drop(self.pool(self.pad(x)))  # feature shape 23 * 39
        x = self.t1(self.conv3(x))  # feature shape 20 * 36
        x = self.drop(self.pool(x))  # feature shape 10 * 18
        x = x.view(-1, 10 * 18 * 96)  # flatten feature maps to 1D
        x = self.t1(self.fc1(x))
        x = self.t2(self.fc2(x))
        return x
```

Be noted that the model takes three layers of kernels to reduce the input size to reasonably-sized 10 by 18 feature maps before being fed to fully connected layers for prediction. The script takes `num_workers` and `use_gpu` parameters that define number of CPUs for batch loading and whether to use GPUs for model training, separately. I ran the model on an instance on the Google Cloud Platform with 1 GPU and 10 CPUs; training the model locally with no GPUs is likely to take too much time to finish.

### STAGE 2.3: Fine tuning model with class weight and optimization hyper parameters

It is worth noting that the project only needs 11 tags for the output prediction, while the training input has more than 30 word tags. I had to relabel most of the word tags into "unknown" to match the desired output. However, after I relabeled the target by replacing non-voice-command words with a single label "unknown", I observed an unbalanced dataset where samples with the "unknown" label is 17.44 times the size of other samples. When trained with this unbalanced data, I noticed that the trained CNN model will simply label all samples with the 'unknown' label to achieve the "best" NLLLoss score, such as below:

```shell
# first row is the prediction for 'unknown'
Confusion Matrix:
[[4080    0    0    0    0    0    0    0    0    0    0]
 [ 248    0    0    0    0    0    0    0    0    0    0]
 [ 264    0    0    0    0    0    0    0    0    0    0]
 [ 238    0    0    0    0    0    0    0    0    0    0]
 [ 219    0    0    0    0    0    0    0    0    0    0]
 [ 243    0    0    0    0    0    0    0    0    0    0]
 [ 230    0    0    0    0    0    0    0    0    0    0]
 [ 239    0    0    0    0    0    0    0    0    0    0]
 [ 225    0    0    0    0    0    0    0    0    0    0]
 [ 256    0    0    0    0    0    0    0    0    0    0]
 [ 231    0    0    0    0    0    0    0    0    0    0]]
```

To deal with the unbalanced data in the training set, I therefore implemented a class weight to penalize the predictions for 'unknown' class so that all labels have an equal weight regardless of the sample size. Hence my weight tensor is defined as `label_weight = torch.FloatTensor([0.057, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])`, where 0.057 equals 1/17.44. After adjusting for the class weights, I received the following confusion matrix early in the training epochs with a NLLoss of 0.41:

```shell
Confusion Matrix:
[[1516 1358  203  111   95   58  497  104   22   22   94]
 [   1  230    6    2    5    1    1    0    2    0    0]
 [   1   96   95    1   63    2    6    0    0    0    0]
 [   5   18    3  175   13    2    5    2    0    2   13]
 [   2   78   18    1  116    1    2    0    0    0    1]
 [   2    9    9    2    0  140   69    0    0   12    0]
 [   2   15    2    1    0    3  204    2    1    0    0]
 [  28    9    8   13    0    0    5  175    0    0    1]
 [   4   50   11    2    0    7    6    0  137    8    0]
 [   4   19   22    3    2   33   30    1    4  138    0]
 [   5   13    1    3    2    0    2    2    0    0  203]]
```

This looks much like what a confusion matrix is 'supposed' to be like: the model is actually learning from the input data now that the target is better defined. I then proceeded to hyper-parameter tuning, starting with a 5-epoch run with the standard **learning rate at 0.1 and momentum at 0.9**, as shown below:

![weighted_lr_0.1_m_0.9](https://github.com/ElvinOuyang/kaggle-tensorflow-speech-recognition-challenge/blob/master/images/training_log_cnn_2017_12_11_15_31_epochs_5.png)

We can see from this chart that the model overshot the global minimum after epoch 2 since the NLLLoss is steadily increasing afterwards. I will need to adjust the hyper-parameter to make the model learn much more slowly. Therefore, in my second 5-epoch run I defined **learning rate as 0.005 and momentum as 0.95**; my result is displayed below:

![weighted_lr_0.005_m_0.95](https://github.com/ElvinOuyang/kaggle-tensorflow-speech-recognition-challenge/blob/master/images/training_log_cnn_2017_12_11_15_51_epochs_5.png)

With a smaller gradient descent step and a larger jump over local minimums, we can clearly see that the model is learning steadily from the input data this time: the NLLLoss is steadily decreasing and the accuracy is steadily increasing. Therefore, I decided that these combination of hyper-parameter is what we desired and went forward to train the data with 30 epochs. My result is displayed below:

![weighted_lr_0.005_m_0.95](https://github.com/ElvinOuyang/kaggle-tensorflow-speech-recognition-challenge/blob/master/images/training_log_cnn_2017_12_11_16_30_epochs_30.png)

Within 30 epochs, we have our highest accuracy at epoch 25 with the confusion matrix as below:

```shell
[Epoch 25] Accuracy: 91.24%, Average Loss: 0.15
Confusion Matrix:
[[3753   30   77   11   22   38   53   56   20   12    8]
 [  13  221    9    1    1    0    1    0    2    0    0]
 [   5    7  231    2   14    0    2    1    1    1    0]
 [  19    0    0  216    0    0    0    1    0    0    2]
 [   7    6   20    1  181    0    0    1    1    2    0]
 [   8    0    1    1    1  215    6    0    1   10    0]
 [  14    0    1    0    0    2  212    0    0    1    0]
 [   8    0    1    1    1    0    0  228    0    0    0]
 [   6    2    2    1    0    4    2    1  200    7    0]
 [   6    1    2    1    0   14    1    0    4  227    0]
 [   9    0    0    0    0    0    0    0    0    0  222]]
```

The best average NLLLoss, however, was achieved at epoch 17 with the confusion matrix as below:

```shell
[Epoch 17] Accuracy: 86.50%, Average Loss: 0.12
Confusion Matrix:
[[3431   47   54   73   60   88   80  110   49   37   51]
 [   9  218    3    0   10    2    2    0    3    0    1]
 [   2    7  218    3   25    2    3    1    1    2    0]
 [   5    0    0  224    2    1    0    1    0    0    5]
 [   5    7   10    3  187    0    0    0    1    4    2]
 [   4    1    0    2    0  222    5    0    1    8    0]
 [   6    1    0    0    0    6  214    0    0    2    1]
 [   3    0    1    3    1    0    0  231    0    0    0]
 [   2    3    1    1    0    6    2    0  205    5    0]
 [   4    1    1    2    0   16    1    0    6  225    0]
 [   2    0    0    2    0    0    1    1    1    0  224]]
```

## STAGE 3: Adding silence audio and resampling the training data
### STAGE 3.1: Adding silence background noise to the training data

Apart from the voice files with spoken words, the competition also provided six "silent" clips that only have background noise. The idea of using background noise in the training process is to let the models learn the difference between any spoken words and bare silence. Therefore, by incorporating silent inputs in the training process, the model should be able to distinguish a unknown spoken word and a silence.

In this stage, I created spectrograms based on the silence recordings and randomly generated clips of the same size as other one second recording. I then save these clips in a directory with a corresponding file map that can be used in the training script.

The resulted training performance is as below:

![silence_trained.png](https://github.com/ElvinOuyang/kaggle-tensorflow-speech-recognition-challenge/blob/master/images/training_log_cnn_2017_12_16_04_10_epochs_30.png)

The model has achieved a similar performance level with the added category for the silence.

### STAGE 3.2: Resampling the training data and adjusting weighting

When analyzing the confusion matrix listed above, I found another potential problem: although the overall accuracy is high, the accuracy for the "positive categories" (i.e. the command words we are trying to classify) are lower than the accuracy for the unknowns and the silences, as shown below:

```shell
# best epoch of the training setting with silence
[Epoch 15] Accuracy: 89.33%, Average Loss: 0.10
Confusion Matrix:
[[3537    1   85   83   28   53   30  130   51   36   42   16]
 [   1  594    0    0    0    0    0    0    0    0    0    0]
 [   3    0  252    5    1    4    0    1    1    0    0    0]
 [   6    0    7  206    0   16    0    0    0    0    1    0]
 [   5    0    2    1  213    3    0    1    2    0    1    4]
 [   7    0    7   14    0  201    1    0    0    0    0    0]
 [   5    0    1    1    0    0  218    6    1    0   11    1]
 [  10    0    5    1    0    0    3  226    0    0    5    1]
 [   4    0    2    1    2    0    0    0  215    0    2    0]
 [   3    0    2    5    0    0    1    0    1  218    2    0]
 [   1    0    1    3    2    0    4    2    1    2  206    1]
 [   6    0    0    0    3    1    1    0    1    0    1  232]]
```

As we can see from this confusion matrix, the accuracies for unknown and silence (and especially for silence) are significantly higher than those command words. No matter how I adjusted the weight penalties for the unknown and silence classes, this situation did not improve. The sheer size of the unwanted "negative categories" are dominating the training input, making the model undertrained with positive categories, even with our weight penalties. The only solution would be to resampling the "unknown" and "silence" categories to have the model train more effectively on the desired categories.

In this stage, I incorporated a random sampling process before the train/test split for each training epoch. Only 2365 (the average size of each command word category) samples of unknown words and silence clips were selected in each training epoch. Since my training samples now have a balanced category distribution, I used the default weighting (i.e. 1 for each category) in the loss calculation function. I received a model performance as below:

![resampled_silence.png](https://github.com/ElvinOuyang/kaggle-tensorflow-speech-recognition-challenge/blob/master/images/training_log_cnn_2017_12_18_16_50_epochs_30.png)

Although the chart might now be clear enough, the model is constantly improving it's performance in the 30 epochs that I ran, instead of reaching a performance plateau within around 15 epochs (which happened in the previous stage). This finding indicates that resampling the training data gave the model a chance to constantly learn from the rich source of unknowns while getting more and more information from the "positive categories". I am definitely on the right track! Below is a quick peek at the confusion matrix of Epoch 30, my best epoch under this setting:

```shell
# best epoch of the training setting with silence and resampling
[Epoch 30] Accuracy: 90.82%, Average Loss: 0.31
Confusion Matrix:
[[224   0   4   4   1   2   3   6   4   0   0   1]
 [  0 225   0   0   0   0   0   0   0   0   0   0]
 [ 13   0 209   4   3   4   0   2   0   0   1   1]
 [  7   0   6 176   3  25   4   0   0   4   4   0]
 [  3   0   1   0 212   0   2   0   1   1   4   0]
 [  6   0   4  13   1 224   1   0   1   1   1   0]
 [  1   0   0   0   1   0 197   1   0   2  11   0]
 [ 14   0   2   0   0   2  19 227   0   0   2   0]
 [  9   0   0   0   2   0   1   0 215   0   0   0]
 [  6   0   3   4   1   0   5   0   0 234   7   1]
 [  2   0   0   2   0   0   8   0   0   3 209   0]
 [  1   0   1   0   3   0   0   0   0   1   0 229]]
```

As indicated in the confusion matrix above, although the total accuracy is on par with previous stages, the predictions for the positive categories (columns 3 to 12) are significantly better than than previous stages. The model is working fine!

## STAGE 4: Stepping in the world of meta-model-stacking

Theoretically speaking, if the model from my previous stage can run on enough number of epochs with the positive samples and the resampled negative samples, it will be able to improve even more. However, since I don't have enough computing resources at my disposal and the training strategy still has a key weakness, I don't expect the model to improve much from current results. The key weakness of my training strategy is the existence of "unknown unknowns": my model works fine when it "hears" words that appeared in the training data (let it be the positive words or all the words that are categorized into "unknown") but will not have a good performance when it hears words that it has never heard of before. This exogenous factor will lead to an "overfitting" situation where the model works perfectly on the validation/test setting that I created in the script, while failing in the real-life situation or the kaggle leaderboard. In fact, [some posts on Kaggle](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/discussion/45682) pointed out that the real challenge indeed lies in the existence of the unheard words with "unknown" labels in the leaderboard test dataset.

In this stage, I will use some meta-model-stacking techniques to address this inherent problem from unknown unknowns.

### STAGE 4.1: Training specialized CNNs for silence and unknowns

Since the unknown unknowns could be a main challenge my model performance, it is important to address the problem with a "specialized checker". Recall that when I predict classes using the log-probability from a Softmax output layer, I determine the class by identifying the class with the highest log-probability (i.e. the class with the highest probability wins the prediction). This method, however, could be a source of errors because in the case of unknown unknowns, the winning class might be winning with a really trivial margin. Therefore, I need to create an **"unknown checker"** that gives a probability of the input being an "unknown" and compare it with the probability from the winning class's probability: if unknown is way more likely than the winning class then I will determine that the input is an "unknown" instead.

In order to create this "unknown checker", I relabeled the target to "unknown", "silence", and "else", where "else" is all the positive labels I used in previous stages. I then trained with the same parameters (learning rate at 0.05 and momentum at 0.95 for SGD) for 30 epochs. The best epoch has a confusion matrix like below:

```shell
# best epoch of unknown checker with silence and weighting
[Epoch 19] Accuracy: 94.06%, Average Loss: 0.1471
Confusion Matrix:
[[3918    1  173]
[   0  595    0]
[ 245    1 2140]]
```

The model gives a good enough performance as a rough "unknown checker". Since I don't need to use the output of this network to make actual predictions, I will stick with this model in my next stage.

### STAGE 4.2: "Stacking" models using logical probability comparison

The final stage in my research is to stack the predictions from main network with the unknown checker, and make a final prediction decision based on the probabilities returned from these two networks. This is mostly a logical decision process where I compare the probability of non-unknown and non-silence predictions: if the probability of the prediction is lower the unknown checker's probability of the input being an unknown, I will revise the prediction to "unknown". However, the revised prediction did not generate a better prediction for the Kaggle public leaderboard. It seems that the network still needs a more aggressive strategy to learn from "unknown unknowns".
