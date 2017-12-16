import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import glob

train_silence_pict_path = '../data/picts/train/silence/'
if not os.path.exists(train_silence_pict_path):
    os.makedirs(train_silence_pict_path)

silence_images = []
for image_path in glob.glob('../data/picts/train/_background_noise_/*.png'):
    image = plt.imread(image_path)[:, :, :3]
    # print(image.shape)
    silence_images.append(image)


silence_spectrograms = []
for image in silence_images:
    index = random.sample(range(len(image) - 99), 1000)
    for i in range(len(index)):
        spec = image[index[i]:(index[i] + 99), :, :]
        silence_spectrograms.append(spec)

"""See a some sample silence spectrograms
plt.figure()
plt.suptitle("Sample Silence Spectrograms")
for i, x in enumerate(random.sample(range(len(silence_spectrograms)), 9)):
    plt.subplot(3, 3, i+1)
    plt.imshow(silence_spectrograms[x], aspect='auto')
    plt.axis('off')
plt.show()
"""

silence_file_map = pd.DataFrame(pd.np.empty((len(silence_spectrograms), 3)),
                                columns=['path', 'target', 'pict'])

silence_file_map['pict'] = silence_file_map.pict.astype(str)
silence_file_map['target'] = 'silence'
silence_file_map['path'] = ''

for i in range(len(silence_spectrograms)):
    img_name = 'silence_%i.png' % (i + 1)
    path = '%s%s' % (train_silence_pict_path, img_name)
    plt.imsave(path, silence_spectrograms[i])
    silence_file_map.pict[i] = path
    if (i + 1) % 500 == 0:
        print(">>> Generating %ith spectrogram..." % (i + 1))
        print(">>> File map %ith row:" % (i + 1))
        print(repr(silence_file_map.iloc[i, ]))

silence_map_csv = '../data/silence_map.csv'
silence_file_map.to_csv(silence_map_csv, index=True)
