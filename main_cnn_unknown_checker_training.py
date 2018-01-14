"""
Script to
1. grab spectrograms using a map file, train 2-D unknown CNN checker with SGD
algorithms
2. also resamples the unknown / silence inputs to have model learn more
frequently from positive inputs (i.e. command words)

Requirements:
spectrogram files generated from the .wav files of the dataset that are stored
in local directory. Subdirectory in the folder is the target label. The data is
extracted to the training model using a map file.
"""

import numpy as np
import pandas as pd
import datetime
import time
from skimage import io, transform
from sklearn.metrics import confusion_matrix
import warnings
from sklearn.model_selection import train_test_split

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# from torch.utils.data.sampler import SequentialSampler
from torchvision import transforms
# from torchnet.meter import AUCMeter

warnings.filterwarnings("ignore")

map_file = '../data/train_map.csv'
silence_map = '../data/silence_map.csv'

model_name = 'unknown_cnn'
batch_size = 250
total_epochs = 30
num_workers = 8
learning_rate = 0.005
momentum = 0.95
load_model = False
last_epoch = 0
use_gpu = True
data_resample = False

# relabel the target to match required output
print(">>> Loading the data mapping files...")
map_df = pd.read_csv(map_file, index_col=0)
silence_df = pd.read_csv(silence_map, index_col=0)
map_df = map_df.append(silence_df, ignore_index=True)
map_df = map_df.loc[map_df.target != '_background_noise_']
targets_to_keep = ['yes', 'no', 'up', 'down', 'left',
                   'right', 'on', 'off', 'stop', 'go', 'silence']
map_df['target'] = map_df['target'].apply(
    lambda x: x if x in targets_to_keep else 'unknown')
# key step: relabel target to have only "unknown", "else", and "silence"
map_df['target'] = map_df['target'].apply(
    lambda x: x if (x == 'unknown' or x == 'silence') else 'else')

label_to_ix = {
    'unknown': 0,
    'silence': 1,
    'else': 2}

map_df['label'] = map_df['target'].apply(lambda x: label_to_ix[x])

print(">>> dataset is geared towards having 'unknown', 'silence', and 'else'")
print(map_df.target.value_counts())
if data_resample:
    label_weight = torch.FloatTensor(
        [1, 1, 1])
    print(">>> this will addressed in data resampling process")
else:
    label_weight = torch.FloatTensor(
        [1, 1, 0.5])
    print(">>> this will be addressed in loss function weighting")

# set random seed
torch.manual_seed(1122)
if use_gpu:
    torch.cuda.manual_seed(1122)


class SPECDataset(Dataset):
    """Spectrogram Dataset"""
    def __init__(self, map_df, transform=None):
        """
        Args:
        map_df: dataframe with columns ['pict', 'label']
        transform: PIL image transformation actions
        """
        self.map_df = map_df
        self.transform = transform

    def __len__(self):
        return self.map_df.shape[0]

    def __getitem__(self, idx):
        pict_path = self.map_df.iloc[idx, 0]
        image = io.imread(pict_path)[:, :, :3]
        target = self.map_df.iloc[idx, 1]
        sample = {'image': image, 'target': target}
        if self.transform:
            sample = self.transform(sample)
        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        return {'image': img, 'target': target}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, target = sample['image'], np.array(
            sample['target'].reshape(1, 1))
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'target': torch.from_numpy(target)}


def generate_Datasets(map_df, test_size, random_state=1122):
    """Create train and validation dataloader based on map_df"""
    x_train, x_test, y_train, y_test = train_test_split(
        map_df.pict, map_df.label, test_size=test_size,
        random_state=random_state)
    map_df_train = pd.DataFrame()
    map_df_train['pict'] = x_train
    map_df_train['label'] = y_train
    map_df_valid = pd.DataFrame()
    map_df_valid['pict'] = x_test
    map_df_valid['label'] = y_test
    trainset = SPECDataset(map_df_train,
                           transform=transforms.Compose([
                               Rescale((99, 161)),
                               ToTensor()]))
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers)

    validset = SPECDataset(map_df_valid,
                           transform=transforms.Compose([
                               Rescale((99, 161)),
                               ToTensor()]))
    validloader = DataLoader(validset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers)
    return trainset, trainloader, validset, validloader


# testsampler = SequentialSampler(testset)
# testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,
#                         sampler=testsampler, num_workers=num_workers)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, 4)
        self.conv2 = nn.Conv2d(24, 48, 3)
        self.conv3 = nn.Conv2d(48, 96, 4)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(10 * 18 * 96, 256)
        self.fc2 = nn.Linear(256, 3)  # change final output to 3 elements
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


print(">>> initiate cnn models")
cnn = CNN()
if use_gpu:
    cnn.cuda()

# load local model if specified
if load_model:
    cnn.load_state_dict(torch.load(load_model))

if use_gpu:
    label_weight = label_weight.cuda()
    print(repr(label_weight))
else:
    print(repr(label_weight))

criterion = nn.NLLLoss(weight=label_weight)
optimizer = optim.SGD(cnn.parameters(), lr=learning_rate, momentum=momentum)


# define a training epoch function
def trainEpoch(dataloader, epoch):
    print(">>> Training Epoch %i" % (epoch + 1))
    cnn.train()
    for i, data in enumerate(dataloader, 0):
        if (i + 1) % 30 == 0:
            print("Training [Epoch %i] Iter %i" % (epoch + 1, i + 1))
        inputs, labels = data['image'], torch.squeeze(data['target'])
        inputs, labels = inputs.float(), labels.long()
        if use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


# define a validate model function
def validateModel(dataloader, epoch):
    print(">>> Validating Epoch %i" % (epoch + 1))
    cnn.eval()
    test_loss = 0
    correct = 0
    pred = np.array([])
    targ = np.array([])
    for data in dataloader:
        inputs, labels = data['image'], torch.squeeze(data['target'])
        if use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs.float()), Variable(labels.long())
        outputs = cnn(inputs)
        test_loss += F.nll_loss(outputs, labels, size_average=False).data[0]
        if use_gpu:
            pred = np.append(pred, outputs.data.topk(1)[1].cpu().numpy())
            targ = np.append(targ, labels.data.cpu().numpy())
        else:
            pred = np.append(pred, outputs.data.topk(1)[1].numpy())
            targ = np.append(targ, labels.data.numpy())
        prd = outputs.data.topk(1)[1]
        if use_gpu:
            correct += prd.eq(labels.data.view_as(prd)).cpu().sum()
        else:
            correct += prd.eq(labels.data.view_as(prd)).sum()
    test_loss /= len(dataloader.dataset)
    test_acc = correct / len(dataloader.dataset)
    cm = confusion_matrix(targ, pred)
    print('[Epoch %i] Accuracy: %.2f%%, Average Loss: %.2f' %
          (epoch + 1, test_acc * 100, test_loss))
    print('Confusion Matrix:')
    print(cm)
    return test_loss, test_acc


def testModel(dataloader, epoch):
    print("Testing Epoch %i" % (epoch + 1))
    cnn.eval()
    pred = np.array([])
    for data in dataloader:
        inputs = data['image'].float()
        if use_gpu:
            inputs = inputs.cuda()
        inputs = Variable(inputs)
        outputs = cnn(inputs)
        if use_gpu:
            pred = np.append(pred,
                             outputs.topk(1)[1].data.view(1, -1).cpu().numpy())
        else:
            pred = np.append(pred,
                             outputs.topk(1)[1].data.view(1, -1).numpy())
    return pred


def resample_data(map_df):
    temp_u = map_df.loc[map_df['target'] == 'unknown'].sample(2365)
    temp_s = map_df.loc[map_df['target'] == 'silence'].sample(2365)
    temp_r = map_df.loc[(map_df['target'] != 'unknown') & (
        map_df['target'] != 'silence')]
    temp = pd.concat([temp_u, temp_s, temp_r], ignore_index=True)
    return temp


# run the training epoch 100 times and test the result
print(">>> training model with cnn")
epoch_loss = []
epoch_acc = []
epoch_time = []
best_loss = 10000
best_acc = 0
for epoch in range(last_epoch, total_epochs):
    start = time.time()
    if data_resample:
        print(">>> resampling training data")
        sample_df = resample_data(map_df)
        trainset, trainloader, validset, validloader = generate_Datasets(
            sample_df, 0.1)
    else:
        trainset, trainloader, validset, validloader = generate_Datasets(
            map_df, 0.1)
    print(">>> train, validation dataset created")
    trainEpoch(trainloader, epoch)
    duration = time.time() - start
    loss, acc = validateModel(validloader, epoch)
    if loss < best_loss:
        best_loss = loss
    if acc > best_acc:
        best_acc = acc
    print(">>> Epoch %i: saving model to local path" % (epoch + 1))
    save_model = '../models/' + model_name + datetime.datetime.now(
    ).strftime("_%Y_%m_%d_%H_%M") + '_epoch_' + str(epoch + 1) + "_lr_" +\
        str(learning_rate) + ".pt"
    torch.save(cnn.state_dict(), save_model)
    print(">>> Best NLLLoss So Far: %.4f" % (best_loss))
    print(">>> Best Accuracy So Far: %.2f%%" % (best_acc * 100))
    epoch_acc.append(acc)
    epoch_loss.append(loss)
    epoch_time.append(duration)


print(">>> outputing epoch performance to local file")
epoch_loss = np.array(epoch_loss)
epoch_acc = np.array(epoch_acc)
epoch_time = np.array(epoch_time)
training_log = pd.DataFrame()
training_log['loss'] = epoch_loss
training_log['accuracy'] = epoch_acc
training_log['duration'] = epoch_time
output_file = '../log/training_log_' + model_name + datetime.datetime.now()\
    .strftime("_%Y_%m_%d_%H_%M") + '_epochs_' + str(total_epochs) + ".csv"
training_log.to_csv(output_file, index=True)
