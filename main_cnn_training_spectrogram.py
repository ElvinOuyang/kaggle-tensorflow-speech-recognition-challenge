"""
Script to grab spectrograms using a map file, train 2-D CNN
model with SGD algorithms, evaluate accuracy and loss on the fly, output
predictions and save models of each epoch for later reference or transfer
learning

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

model_name = 'cnn'
batch_size = 250
total_epochs = 30
num_workers = 3
learning_rate = 1
momentum = 0.9
load_model = False
last_epoch = 0
use_gpu = False

# relabel the target to match required output
print(">>> Loading the data mapping file...")
map_df = pd.read_csv(map_file, index_col=0)
targets_to_keep = ['yes', 'no', 'up', 'down', 'left',
                   'right', 'on', 'off', 'stop', 'go', 'silence']
map_df['target'] = map_df['target'].apply(
    lambda x: x if x in targets_to_keep else 'unknown')
label_to_ix = {
    'unknown': 0,
    'down': 1,
    'go': 2,
    'left': 3,
    'no': 4,
    'off': 5,
    'on': 6,
    'right': 7,
    'stop': 8,
    'up': 9,
    'yes': 10}

"""
for x in map_df.target.unique():
    if x not in label_to_ix.keys():
        label_to_ix[x] = len(label_to_ix)
"""

map_df['label'] = map_df['target'].apply(lambda x: label_to_ix[x])
print(map_df.head())

print(">>> dataset is heavily unbalanced towards 'unknown' tag")
print(map_df.target.value_counts())
print(">>> this will be addressed in loss function weighting")
"""calculate the corresponding weight needed for each class
w = np.array(
    [41045, 2380, 2377, 2375, 2375, 2372, 2367, 2367, 2359, 2357, 2353])
weighted = w / (w.min())
weighted
"""
label_weight = torch.FloatTensor([1, 17.44, 17.44, 17.44, 17.44, 17.44, 17.44,
                                  17.44, 17.44, 17.44, 17.44])

# create train and validation set based on map_df
x_train, x_test, y_train, y_test = train_test_split(
    map_df.pict, map_df.label, test_size=0.1, random_state=1122)

map_df_train = pd.DataFrame()
map_df_train['pict'] = x_train
map_df_train['label'] = y_train

map_df_valid = pd.DataFrame()
map_df_valid['pict'] = x_test
map_df_valid['label'] = y_test

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


""" test the SPECDataset class
pict_path = map_df_train.iloc[10, 0]
image = io.imread(pict_path)[:, :, :3]
target = map_df_train.iloc[10, 1]

spec_dataset = SPECDataset(map_df_train)
fig = plt.figure()
for i in range(4):
    sample = spec_dataset[i]
    ax = plt.subplot(2, 2, i + 1)
    plt.tight_layout()
    ax.set_title("Sample #{}".format(i))
    ax.axis('off')
    plt.imshow(sample['image'])
    if i == 3:
        plt.show()
        break

print(sample['image'].shape)
print(sample['target'])
"""


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

"""
rescale = Rescale((99, 161))
sample_rescaled = rescale(sample)
sample_rescaled['target']
sample_rescaled['image'].shape
"""


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, target = sample['image'], np.array(
            sample['target'].reshape(1, 1))
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'target': torch.from_numpy(target)}
"""
totensor = ToTensor()
sample_tensor = totensor(sample_rescaled)
sample_tensor['image'].shape
sample_tensor['target']
"""

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


"""
for i_batch, sample_batched in enumerate(trainloader):
    print(i_batch, sample_batched['image'].size(),
        sample_batched['target'].size())
    if i_batch == 3:
        break
"""
# testsampler = SequentialSampler(testset)
# testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,
#                         sampler=testsampler, num_workers=num_workers)
print(">>> train, validation dataset created")

"""
Quick test trainloader see if it is working
for x in trainloader:
    x = x
    print(repr(x))
    break
"""


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


print(">>> initiate cnn models")
cnn = CNN()
if use_gpu:
    cnn.cuda()

# load local model if specified
if load_model:
    cnn.load_state_dict(torch.load(load_model))

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
        test_loss += F.nll_loss(outputs, labels, weight=label_weight,
                                size_average=False).data[0]
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


"""quick test run code
test_loss = 0
pred = np.array([])
targ = np.array([])
correct = 0
for data in trainloader:
    data = data
    break
inputs, labels = data['image'], torch.squeeze(data['target'])
inputs, labels = Variable(inputs.float()), Variable(labels.long())
outputs = cnn(inputs)
test_loss += F.nll_loss(outputs, labels, weight=label_weight,
                        size_average=False).data[0]
pred = np.append(pred, outputs.data.topk(1)[1].numpy())
targ = np.append(targ, labels.data.numpy())
prd = outputs.data.topk(1)[1]
correct += prd.eq(labels.data.view_as(prd)).sum()
cm = confusion_matrix(targ, pred)
from collections import Counter
Counter(targ)
Counter(pred)
test_loss /= batch_size
test_acc = correct / batch_size
print('[Epoch %i] Accuracy: %.2f percent, Average Loss: %.2f' %
      (1, test_acc * 100, test_loss))
"""


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


# run the training epoch 100 times and test the result
print(">>> training model with cnn")
epoch_loss = []
epoch_acc = []
epoch_time = []
best_acc = 0
for epoch in range(last_epoch, total_epochs):
    start = time.time()
    trainEpoch(trainloader, epoch)
    duration = time.time() - start
    loss, acc = validateModel(validloader, epoch)
    if acc > best_acc:
        best_acc = acc
        save_model = '../models/' + model_name + datetime.datetime.now(
        ).strftime("_%Y_%m_%d_%H_%M") + '_epoch_' + str(epoch + 1) + "_lr_" +\
            str(learning_rate) + ".pt"
        print(">>> Epoch %i: saving model to local path" % (epoch + 1))
        torch.save(cnn.state_dict(), save_model)
    print(">>> Best Accuracy So Far: %.4f%%" % (best_acc * 100))
    epoch_acc.append(acc)
    epoch_loss.append(loss)
    epoch_time.append(duration)

print(">>> outputing predictions to local file")
epoch_loss = np.array(epoch_loss)
epoch_acc = np.array(epoch_acc)
epoch_time = np.array(epoch_time)
training_log = pd.DataFrame()
training_log['loss'] = epoch_loss
training_log['accuracy'] = epoch_acc
training_log['duration'] = epoch_time
output_file = '../log/training_log_' + model_name + datetime.datetime.now(
    ).strftime("_%Y_%m_%d_%H_%M") + '_epochs_' + str(total_epochs) + ".csv"
training_log.to_csv(output_file, index=True)

# TODO: resample model for a balanced data structure
