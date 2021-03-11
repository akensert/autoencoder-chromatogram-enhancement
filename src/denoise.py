import numpy as np
import tensorflow as tf
import pandas as pd

from signal_processing import apply_interpolation

tf.config.set_visible_devices([], 'GPU')

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='../data/chromatogram_ISO.csv')
args = parser.parse_args()

data = pd.read_csv(args.path, header=None)
time, signal = data.iloc[:, 0].values, data.iloc[:, 1].values
time, signal = apply_interpolation(time, signal, 8192)

model = tf.saved_model.load('../output/model/')

smooth_signal = model.smooth(signal)

pd.DataFrame(np.stack([time, smooth_signal], axis=1)).to_csv(
    args.path + '.smooth.csv', header=None, index=False)
