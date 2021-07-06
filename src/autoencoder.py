import tensorflow as tf


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
