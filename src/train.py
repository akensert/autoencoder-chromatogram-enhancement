import numpy as np
import tensorflow as tf
import math


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


DECAY = 1e-4

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
                    kernel_regularizer=tf.keras.regularizers.l2(DECAY),
                    name='conv1d_downsample_{}'.format(i),
                    padding='same'))
            # self.encoder.add(tf.keras.layers.Dropout(0.2))

        # Define Decoder
        decoder_input_shape = (input_shape[0]//(stride_size**len(layer_sizes)), layer_sizes[-1])
        self.decoder1 = tf.keras.Sequential()
        self.decoder1.add(tf.keras.layers.Input(shape=decoder_input_shape))
        for i, (filter_size, kernel_size) in enumerate(zip(reversed(layer_sizes), reversed(kernel_sizes))):

            self.decoder1.add(
                tf.keras.layers.Conv1DTranspose(
                    filters=filter_size,
                    kernel_size=kernel_size,
                    strides=stride_size,
                    activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(DECAY),
                    name='conv1d_upsample_{}_1'.format(i),
                    padding='same'))
            # self.decoder1.add(tf.keras.layers.Dropout(0.2))

        self.decoder1.add(
            tf.keras.layers.Conv1DTranspose(
                filters=1,
                kernel_size=1,
                strides=1,
                activation='linear',
                name='conv1d_upsample_{}_1'.format(i+1),
                padding='same'))

        self.decoder2 = tf.keras.Sequential()
        self.decoder2.add(tf.keras.layers.Input(shape=decoder_input_shape))
        for i, (filter_size, kernel_size) in enumerate(zip(reversed(layer_sizes), reversed(kernel_sizes))):

            self.decoder2.add(
                tf.keras.layers.Conv1DTranspose(
                    filters=filter_size,
                    kernel_size=kernel_size,
                    strides=stride_size,
                    activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(DECAY),
                    name='conv1d_upsample_{}_2'.format(i),
                    padding='same'))
            # self.decoder2.add(tf.keras.layers.Dropout(0.2))

        self.decoder2.add(
            tf.keras.layers.Conv1DTranspose(
                filters=1,
                kernel_size=1,
                strides=1,
                activation='linear',
                kernel_regularizer=tf.keras.regularizers.l2(DECAY),
                name='conv1d_upsample_{}_2'.format(i+1),
                padding='same'))

        self.encoder._name = 'Encoder'
        self.decoder1._name = 'Decoder1'
        self.decoder2._name = 'Decoder2'

    @property
    def trainable_weights(self):
        return (
            self.encoder.trainable_weights +
            self.decoder1.trainable_weights +
            self.decoder2.trainable_weights
        )

    def train_step(self, data):
        x, (y, y_der) = data
        with tf.GradientTape() as tape:
            y_pred, y_der_pred = self(x, training=True)
            loss1 = self.compiled_loss(y, y_pred)
            loss2 = self.compiled_loss(y_der, y_der_pred)
            loss = loss1*0.5 + loss2*0.5
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, training=False):
        x = self.encoder(inputs, training=training)
        x1 = self.decoder1(x, training=training)
        x2 = self.decoder2(x, training=training)
        return x1, x2

    @tf.function(input_signature=[
        tf.TensorSpec([8192], tf.float32),
    ])
    def smooth(self, chromatogram):
        x = chromatogram[tf.newaxis, :, tf.newaxis]
        x = self.encoder(x, training=False)
        x = self.decoder1(x, training=False)
        return tf.squeeze(x)

    @tf.function(input_signature=[
        tf.TensorSpec([8192], tf.float32),
    ])
    def smooth_der(self, chromatogram):
        x = chromatogram[tf.newaxis, :, tf.newaxis]
        x = self.encoder(x, training=False)
        x = self.decoder2(x, training=False)
        return tf.squeeze(x)

    def summary(self):
        self.encoder.summary()
        self.decoder1.summary()
        self.decoder2.summary()

if __name__ == '__main__':
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU', type=bool, default=True)
    parser.add_argument('--outpath', type=str, default='../output/model')
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


    train_path = '../input/simulations/train/'
    layer_sizes = [256, 128, 128, 64, 64, 32]
    kernel_sizes = [9]*len(layer_sizes)

    generator = Generator(
        path=train_path,
        batch_size=32
    )
    cae = CAE(
        layer_sizes=layer_sizes,
        kernel_sizes=kernel_sizes,
        stride_size=2,
        input_shape=(8192, 1)
    )
    callback = tf.keras.callbacks.LearningRateScheduler(
        schedule=lambda epoch, lr: lr * 0.8 if epoch > 0 else lr, verbose=1
    )

    cae.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='MSE')
    cae.fit(
        generator,
        epochs=10,
        callbacks=[callback],
        use_multiprocessing=True,
        workers=4,
        verbose=1
    )

    tf.saved_model.save(cae, export_dir=args.outpath)
    print("model saved to {}".format(args.outpath))

    # #DEBUG
    # for i in range(12):
    #     cae.fit(
    #         generator,
    #         epochs=1,
    #         # use_multiprocessing=True,
    #         # workers=4,
    #         verbose=1
    #     )
    #     cae.optimizer.lr = cae.optimizer.lr * 0.8
    #     tf.saved_model.save(cae, export_dir=args.outpath+f'{i}')
    #     os.system('python evaluate.py --T=white --N=500 --inpath={}'.format(args.outpath+f'{i}'))
    #     print("model saved to {}".format(args.outpath))
