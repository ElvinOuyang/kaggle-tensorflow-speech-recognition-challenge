import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os
import numpy as np
import pandas as pd


# set up file paths
train_audio_path = '../data/train/audio/'
train_pict_path = '../data/picts/train/'
train_map_csv = '../data/train_map.csv'
test_pict_path = '../data/picts/test/'
test_audio_path = '../data/test/audio/'
test_map_csv = '../data/test_map.csv'

# set up paths
if not os.path.exists(train_pict_path):
    os.makedirs(train_pict_path)

if not os.path.exists(test_pict_path):
    os.makedirs(test_pict_path)

# get all subfolder list
subfolderlist = []
for x in os.listdir(train_audio_path):
    if os.path.isdir(train_audio_path + '/' + x):
        subfolderlist.append(x)
        if not os.path.exists(train_pict_path + '/' + x):
            os.makedirs(train_pict_path + '/' + x)

# obtain a file map with path and target for training iterators
file_target = []
for x in subfolderlist:
    temp = [['/'.join([train_audio_path[:-1], x, y]), x] for
            y in os.listdir(train_audio_path + x) if '.wav' in y]
    file_target += temp

train_file_map = pd.DataFrame(file_target, columns=['path', 'target'])
print(">>> Generated train_file_map dataframe:")
print(train_file_map.head())

# generate a test_file_map
test_path = [[x, 0] for x in os.listdir(test_audio_path) if '.wav' in x]
test_file_map = pd.DataFrame(test_path, columns=['path', 'target'])
print(">>> Generated test_file_map dataframe:")
print(test_file_map.head())

# Function that changes .wav file into a spectrogram numbers


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

'''
# Visualize several files
fig = plt.figure(figsize=(20, 20))
locs = [0, 3000, 5000, 7000, 9000, 11000, 13000, 15000, 17000]

for i, loc in enumerate(locs):
    plt.subplot(3, 3, i+1)
    # pull the labels
    label = train_file_map.iloc[loc, ].target
    plt.title(label)
    # create spectogram
    samplerate, test_sound = wavfile.read(train_file_map.iloc[i, ].path)
    _, spectrogram = log_specgram(test_sound, samplerate)
    plt.imshow(spectrogram.T, aspect='auto', origin='lower')
    plt.axis('off')
plt.savefig('images/spectrogram_samples.png', bbox_inches='tight')
'''


def wav2img(wav_path, targetdir='', figsize=(4, 4)):
    """
    takes in wave file path
    and the fig size. Default 4,4 will make images 288 x 288
    """
    fig = plt.figure(figsize=figsize)
    # use soundfile library to read in the wave files
    samplerate, test_sound = wavfile.read(wav_path)
    _, spectrogram = log_specgram(test_sound, samplerate)
    # create output path
    output_file = wav_path.split('/')[-1].split('.wav')[0]
    output_file = targetdir + '/' + output_file
    # plt.imshow(spectrogram.T, aspect='auto', origin='lower')
    plt.imsave('%s.png' % output_file, spectrogram)
    plt.close()
    return (output_file + '.png')


# generate train pictures
train_file_map['pict'] = ''
print(">>> Creating spectrograms for train .wav files")
for i in range(train_file_map.shape[0]):
    targetdir = train_pict_path + train_file_map.target[i]
    pict_path = wav2img(train_file_map.path[i], targetdir=targetdir)
    train_file_map.pict[i] = pict_path
    if (i + 1) % 500 == 0:
        print(">>> Generating %ith spectrogram..." % (i + 1))
        print(">>> File map %ith row:" % (i + 1))
        print(repr(train_file_map.iloc[i, ]))

'''
i = 0
targetdir = train_pict_path + train_file_map.target[i]
targetdir
pict_path = wav2img(train_file_map.path[i], targetdir=targetdir)
pict_path
train_file_map.pict[i] = pict_path
print(repr(train_file_map.iloc[i,]))
'''

# saving train map file
train_file_map.to_csv(train_map_csv, index=True)

# generate test pictures
test_file_map['pict'] = ''
print(">>> Creating spectrograms for test .wav files")
for i in range(test_file_map.shape[0]):
    targetdir = test_pict_path[:-1]
    pict_path = wav2img(test_file_map.path[i], targetdir=targetdir)
    test_file_map.pict[i] = pict_path
    if (i + 1) % 500 == 0:
        print(">>> Generating %ith spectrogram..." % (i + 1))
        print(">>> File map %ith row:" % (i + 1))
        print(repr(test_file_map.iloc[i, ]))

# saving test map file
test_file_map.to_csv(test_map_csv, index=True)
