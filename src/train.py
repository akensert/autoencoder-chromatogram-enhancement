import numpy as np
import math
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import tqdm


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
        for i, idx in enumerate(batch_indices):
            _, x, y = np.load(self.path+f'chromatogram_{idx}.npy')
            batch_x[i,] = x[:, np.newaxis]
            batch_y[i,] = y[:, np.newaxis]

        return np.array(batch_x), np.array(batch_y)


class CAE(tf.keras.Model):

    def __init__(self,
                 layer_sizes,
                 kernel_sizes,
                 stride_size,
                 input_shape,
                 name='autoencoder',
                 **kwargs):
        super(CAE, self).__init__(name=name, **kwargs)

        # Define Encoder
        self.encoder = tf.keras.Sequential()
        self.encoder.add(tf.keras.layers.Input(shape=input_shape))
        for i, (filter_size, kernel_size) in enumerate(zip(layer_sizes, kernel_sizes)):
            # print(filter_size, kernel_size)
            self.encoder.add(
                tf.keras.layers.Conv1D(
                    filters=filter_size,
                    kernel_size=kernel_size,
                    strides=stride_size,
                    activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                    name='conv1d_downsample_{}'.format(i),
                    padding='same'))

        # Define Decoder
        decoder_input_shape = (input_shape[0]//(stride_size**len(layer_sizes)), layer_sizes[-1])
        self.decoder = tf.keras.Sequential()
        self.decoder.add(tf.keras.layers.Input(shape=decoder_input_shape))
        for i, (filter_size, kernel_size) in enumerate(zip(reversed(layer_sizes), reversed(kernel_sizes))):

            self.decoder.add(
                tf.keras.layers.Conv1DTranspose(
                    filters=filter_size,
                    kernel_size=kernel_size,
                    strides=stride_size,
                    activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                    name='conv1d_upsample_{}'.format(i),
                    padding='same'))

        self.decoder.add(
            tf.keras.layers.Conv1DTranspose(
                filters=1,
                kernel_size=1,
                strides=1,
                activation='linear',
                name='conv1d_upsample_{}'.format(i+1),
                padding='same'))

        self.encoder._name = 'Encoder'
        self.decoder._name = 'Decoder'

        self.mae_loss = tf.keras.losses.MeanAbsoluteError()
        self.mae = tf.keras.metrics.Mean()

    @property
    def trainable_weights(self):
        return (
            self.encoder.trainable_weights +
            self.decoder.trainable_weights
        )

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            x = self(x, training=True)
            loss = self.mae_loss(y, x)
            self.mae.update_state(loss)
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        return {
            'MAE_loss': self.mae.result(),
        }

    def call(self, inputs, training=False):
        x = self.encoder(inputs, training=training)
        x = self.decoder(x, training=training)
        return x

    @tf.function(input_signature=[
        tf.TensorSpec([8192], tf.float32),
    ])
    def smooth(self, chromatogram):
        x = x[tf.newaxis, :, tf.newaxis]
        x = self.encoder(x, training=False)
        x = self.decoder(x, training=False)
        return tf.squeeze(x)

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU', type=bool, default=True)
    args = parser.parse_args()

    if args.GPU:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        num_gpus = len(gpus)
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(num_gpus, "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                print(e)
    else:
        # if GPU exist, make it invisible
        tf.config.set_visible_devices([], 'GPU')


    generator = Generator(
        path='../input/simulations/train/',
        batch_size=32)

    cae = CAE(
        layer_sizes=[128, 128, 64, 64, 32],
        kernel_sizes=[7, 7, 7, 7, 7, 7],
        stride_size=2,
        input_shape=(8192, 1)
        )
    callback = tf.keras.callbacks.LearningRateScheduler(
        schedule=lambda epoch, lr: lr * 0.5**epoch
    )

    cae.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    cae.fit(generator, epochs=4, verbose=1, callbacks=[callback])

    tf.saved_model.save(cae, export_dir='../output/model')
    print("model saved to '../output/'")
