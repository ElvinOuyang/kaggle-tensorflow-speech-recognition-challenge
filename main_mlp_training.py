"""
Script to connect to sqlite db using data streaming iterators, train mlp
model with SGD algorithms, evaluate accuracy and loss on the fly, output
predictions and save models of each epoch for later reference or transfer
learning

Requirements:
mfcc files generated from the .wav files of the dataset that are centered
and scaled. The 1-13 columns are MFCCs, 14-26 columns are deltas, 27-39 columns
are delta-deltas.
"""

import numpy as np
import pandas as pd
import datetime
import time
from sqlalchemy import create_engine
from sklearn.metrics import confusion_matrix


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler
# from torchnet.meter import AUCMeter

sqlite_url = 'sqlite:///../data/mfcc_phonemes.db'
training_tables = ['y_train', 'x_train']
testing_tables = ['y_test', 'x_test']
batch_size = 250
total_epochs = 30
num_workers = 0
learning_rate = 0.1
momentum = 0.9
load_model = False
last_epoch = 0
num_inputs = 13
model_name = 'mlp'

# set random seed
torch.manual_seed(1122)

# create linkage to database
print(">>> link to sqlite database kkbox.db")
mpdb_engine = create_engine(sqlite_url)
mpdb_conn = mpdb_engine.connect()

# get total label tags
# test = mpdb_conn.execute("SELECT DISTINCT phoneme FROM y_train").fetchall()
# test = np.array([t[0] for t in test])
# len(test)


class MPDBDataset(Dataset):
    """MPDB Final Set (SQLITE) for Training and Testing."""
    def __init__(self, sql_conn, table_names, num_inputs):
        """
        Args:
        sql_conn: dabase connection used by sqlalchemy to build engine
        table_name: respective table to query for
        """
        self.sql_conn = sql_conn
        self.table_names = table_names
        self.num_inputs = num_inputs

    def __len__(self):
        ln = self.sql_conn.execute(
            'SELECT max(rowid) FROM ' + self.table_names[0]).scalar()
        return ln

    def __getitem__(self, idx):
        stmt_y = 'SELECT * FROM ' + self.table_names[0] + ' WHERE rowid = ' +\
            str(idx + 1)
        line_y = self.sql_conn.execute(stmt_y).fetchone()
        line_y = [y for y in line_y]
        stmt_x = 'SELECT * FROM ' + self.table_names[1] + ' WHERE rowid = ' +\
            str(idx + 1)
        line_x = self.sql_conn.execute(stmt_x).fetchone()
        line_x = [x for x in line_x][:self.num_inputs]
        line = line_y + line_x
        line = np.asarray(line, dtype=float)
        line[np.isnan(line)] = 0
        return line


trainset = MPDBDataset(mpdb_conn, training_tables, num_inputs)
testset = MPDBDataset(mpdb_conn, testing_tables, num_inputs)
trainloader = DataLoader(trainset, batch_size=batch_size,
                         shuffle=True, num_workers=num_workers)
testsampler = SequentialSampler(testset)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                        sampler=testsampler, num_workers=num_workers)
print(">>> train, test dataset created")

"""
Quick test trainloader see if it is working
for x in trainloader:
    print(repr(x))
    break
"""


class MLP(nn.Module):
    def __init__(self, num_inputs):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(num_inputs, 50)
        self.t1 = nn.Tanh()
        self.l2 = nn.Linear(50, 38)
        self.t2 = nn.LogSoftmax()

    def forward(self, x):
        x = self.t1(self.l1(x))
        x = self.t2(self.l2(x))
        return x


print(">>> initiate mlp models")
mlp = MLP(num_inputs)
# load local model if specified
if load_model:
    mlp.load_state_dict(torch.load(load_model))

criterion = nn.NLLLoss()
optimizer = optim.SGD(mlp.parameters(), lr=learning_rate, momentum=momentum)


# define a training epoch function
def trainEpoch(dataloader, epoch):
    print(">>> Training Epoch %i" % (epoch + 1))
    mlp.train()
    for i, data in enumerate(dataloader, 0):
        if (i + 1) % 100 == 0:
            print("Training [Epoch %i] Iter %i" % (epoch + 1, i + 1))
        inputs, labels = data[:, 1:], data[:, 0]
        inputs, labels = inputs.float(), labels.long()
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = mlp(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def validateModel(dataloader, epoch):
    print(">>> Validating Epoch %i" % (epoch + 1))
    mlp.eval()
    test_loss = 0
    correct = 0
    pred = np.array([])
    targ = np.array([])
    for data in dataloader:
        inputs, labels = data[:, 1:], data[:, 0]
        inputs, labels = Variable(inputs.float()), Variable(labels.long())
        outputs = mlp(inputs)
        test_loss += F.nll_loss(outputs, labels, size_average=False).data[0]
        pred = np.append(pred, outputs.data.topk(1)[1].numpy())
        targ = np.append(targ, labels.data.numpy())
        prd = outputs.data.topk(1)[1]
        correct += prd.eq(labels.data.view_as(prd)).sum()
    test_loss /= len(dataloader.dataset)
    test_acc = correct / len(dataloader.dataset)
    cm = confusion_matrix(targ, pred)
    print('[Epoch %i] Accuracy: %.2f%%, Average Loss: %.2f' %
          (epoch + 1, test_acc * 100, test_loss))
    print('Confusion Matrix:')
    print(cm)
    return test_loss, test_acc


"""
for data in trainloader:
    data = data
    break
inputs, labels = data[:, 1:], data[:, 0]
inputs, labels = Variable(inputs.float()), Variable(labels.long())
outputs = mlp(inputs)
test_loss += F.nll_loss(outputs, labels, size_average=False).data[0]
pred = np.append(pred, outputs.data.topk(1)[1].numpy())
targ = np.append(targ, labels.data.numpy())
prd = outputs.data.topk(1)[1]
correct += prd.eq(labels.data.view_as(prd)).sum()
cm = confusion_matrix(targ, pred)
test_loss /= len(trainloader.dataset)
test_acc = correct / len(trainloader.dataset)
print('[Epoch %i] Accuracy: %.2f percent, Average Loss: %.2f' %
      (1, test_acc * 100, test_loss))
"""


def testModel(dataloader, epoch):
    print("Testing Epoch %i" % (epoch + 1))
    mlp.eval()
    pred = np.array([])
    for data in dataloader:
        inputs = data[:, 1:].float()
        inputs = Variable(inputs)
        outputs = mlp(inputs)
        pred = np.append(pred,
                         outputs.topk(1)[1].data.view(1, -1).numpy())
    return pred


# run the training epoch 100 times and test the result
print(">>> training model with mlp")
epoch_loss = []
epoch_acc = []
epoch_time = []
best_acc = 0
for epoch in range(last_epoch, total_epochs):
    start = time.time()
    trainEpoch(trainloader, epoch)
    duration = time.time() - start
    loss, acc = validateModel(testloader, epoch)
    if acc > best_acc:
        best_acc = acc
        save_model = '../models/' + model_name + datetime.datetime.now(
        ).strftime("_%Y_%m_%d_%H_%M") + '_epoch_' + str(epoch + 1) + "_lr_" +\
            str(learning_rate) + ".pt"
        print(">>> Epoch %i: saving model to local path" % (epoch + 1))
        torch.save(mlp.state_dict(), save_model)
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
