"""
Previous Scripts (stacking 1 & 2)
1. grab spectrograms using a map file
2. load saved models (main and unknown)
3. generate winning-class and its probability
4. generate unknown probability

This Script (stacking 3)
5. make logical decisions based on step 3 and 4
6. assemble final prediction

Requirements:
spectrogram files generated from the .wav files of the dataset that are stored
in local directory. Subdirectory in the folder is the target label. The data is
extracted to the training model using a map file.
"""

import numpy as np
import pandas as pd
import datetime
# define the probability buff to add to the main class probability
# before comparing with unknown check's unknown probability
prob_buff = 0.3

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

print(">>> Loading model predictions and probabilities...")
test_1 = pd.read_csv('../stacking/test_1__2018_01_14_20_10.csv')
test_2 = pd.read_csv('../stacking/test_2__2018_01_14_21_05.csv')

test_1['pred'] = test_1.pred.astype(int)
test_2['pred'] = test_2.pred.astype(int)
test_1.head()
test_2.head()

final_pred = np.array([])
for i in range(test_1.shape[0]):
    if test_1.pred[i] == 0 or test_1.pred[i] == 1:
        final_pred = np.append(final_pred, test_1.pred[i])
    else:
        if (test_1.pred_prob[i] + prob_buff) < test_2.unknown_prob[i]:
            final_pred = np.append(final_pred, 0)
        else:
            final_pred = np.append(final_pred, test_1.pred[i])
final_pred = final_pred.astype(int)

'''
i = 2
if test_1.pred[i] == 0 or test_1.pred[i] == 1:
    final_pred = np.append(final_pred, test_1.pred[i])
else:
    print(i)
    if (test_1.pred_prob[i] + prob_buff) < test_2.unknown_prob[i]:
        final_pred = np.append(final_pred, 0)
    else:
        final_pred = np.append(final_pred, test_1.pred[i])
final_pred
'''


def ix_to_label(label_to_ix, pred):
    for key, value in label_to_ix.items():
        if pred == value:
            label = key
        else:
            pass
    return label

submission = pd.DataFrame()
submission['pred'] = final_pred
submission['label'] = submission.pred.apply(
    lambda x: ix_to_label(label_to_ix, x))
submission['fname'] = test_1.fname
submission = submission.drop(['pred'], axis=1)
print(submission.head())
submission_name = '../log/submission_stacked_prob_buff_' + str(prob_buff) +\
    datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M") + ".csv"
submission.to_csv(submission_name, index=False)
