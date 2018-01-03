"""
Script to
1. grab spectrograms using a map file
2. Load saved model
3. generate results based on trained model

Requirements:
spectrogram files generated from the .wav files of the dataset that are stored
in local directory. Subdirectory in the folder is the target label. The data is
extracted to the training model using a map file.
"""

import numpy as np
import pandas as pd
from skimage import io, transform
import datetime
import warnings

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler
from torchvision import transforms

warnings.filterwarnings("ignore")
test_map = '../data/test_map.csv'

model_name = 'cnn'
batch_size = 250
num_workers = 10
load_model = '../models/cnn_2017_12_18_16_50_epoch_30_lr_0.005.pt'
use_gpu = True

print(">>> Loading the data mapping files...")
map_df = pd.read_csv(test_map, index_col=0)
label_to_ix = {
    'unknown': 0,
    'silence': 1,
    'down': 2,
    'go': 3,
    'left': 4,
    'no': 5,
    'off': 6,
    'on': 7,
    'right': 8,
    'stop': 9,
    'up': 10,
    'yes': 11}

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

map_df_test = pd.DataFrame()
map_df_test['pict'] = map_df.pict
map_df_test['label'] = map_df.target.astype(int)
testset = SPECDataset(map_df_test,
                      transform=transforms.Compose([
                          Rescale((99, 161)),
                          ToTensor()]))
testsampler = SequentialSampler(testset)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                        sampler=testsampler, num_workers=num_workers)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, 4)
        self.conv2 = nn.Conv2d(24, 48, 3)
        self.conv3 = nn.Conv2d(48, 96, 4)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(10 * 18 * 96, 256)
        self.fc2 = nn.Linear(256, 12)
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


def testModel(dataloader):
    cnn.eval()
    pred = np.array([])
    for i, data in enumerate(dataloader):
        if (i + 1) % 20 == 0:
            print(">>> Generating iteration %i..." % (i + 1))
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

print(">>> Generating testing results...")
submission = pd.DataFrame()
submission['pred'] = testModel(testloader)


def ix_to_label(label_to_ix, pred):
    for key, value in label_to_ix.items():
        if pred == value:
            label = key
        else:
            pass
    return label

submission['label'] = submission.pred.apply(
    lambda x: ix_to_label(label_to_ix, x))
submission['fname'] = map_df.path.apply(lambda x: x.split('/')[-1])
submission = submission.drop(['pred'], axis=1)
print(submission.head())
submission_name = '../log/submission_' + datetime.datetime.now().\
    strftime("_%Y_%m_%d_%H_%M") + ".csv"
submission.to_csv(submission_name, index=False)
