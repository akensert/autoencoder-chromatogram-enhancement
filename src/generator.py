import tensorflow as tf
import numpy as np


class Generator(tf.keras.utils.Sequence):

    def __init__(self, path, batch_size, num_examples=190_000, random_seed=42):
        self.path = path
        self.batch_size = batch_size
        self.num_examples = num_examples
        self.random_seed = random_seed
        self.on_epoch_end()

    def __len__(self):
        return self.num_examples // self.batch_size

    def on_epoch_end(self):
        np.random.seed(self.random_seed)
        self.random_seed += 1
        self.indices = np.arange(self.num_examples)
        np.random.shuffle(self.indices)

    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_x = np.empty([self.batch_size, 8192, 1])
        batch_y = np.empty([self.batch_size, 8192, 1])
        batch_y_der = np.empty([self.batch_size, 8192, 1])
        for i, idx in enumerate(batch_indices):
            _, x, y, y_der = np.load(self.path+f'chromatogram_{idx}.npy')
            batch_x[i,] = x[:, np.newaxis]
            batch_y[i,] = y[:, np.newaxis]
            batch_y_der[i,] = y_der[:, np.newaxis]
        return np.array(batch_x), (np.array(batch_y), np.array(batch_y_der))
