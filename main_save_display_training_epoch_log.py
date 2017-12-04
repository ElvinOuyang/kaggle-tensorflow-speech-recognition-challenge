import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

log_path = "../log/training_log2017_12_04_04_29_epochs_100.csv"
my_dpi = 90
out_path = "../log/training_log2017_12_04_04_29_epochs_100.png"
model_name = "MFCC-Phonemes MLP (Alpha: 0.1, Momentum: 0.9)"

df = pd.read_csv(log_path, index_col=0)

plt.figure(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)
plt.suptitle(model_name + " Training Log")
plt.subplot(311)
plt.plot((df.index + 1), df.duration)
plt.ylabel('Time (Seconds)')
plt.tick_params(
    axis='x',
    which='both',
    bottom='off',
    top='off',
    labelbottom='off')

plt.subplot(312)
plt.plot((df.index + 1), df.accuracy * 100)
plt.ylabel('Accuracy (%)')
plt.tick_params(
    axis='x',
    which='both',
    bottom='off',
    top='off',
    labelbottom='off')

plt.subplot(313)
plt.plot((df.index + 1), df.loss)
plt.ylabel('NLLLoss')
plt.xlabel('Epochs')
plt.savefig(out_path, bbox_inches='tight', dpi=my_dpi)
