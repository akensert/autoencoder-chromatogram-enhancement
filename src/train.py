import numpy as np
import tensorflow as tf
import os
import argparse

from generator import Generator
from autoencoder import CAE


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU', type=str, default='true')
    parser.add_argument('--outpath', type=str, default='../output/model')
    args = parser.parse_args()

    if args.GPU.lower() == 'true':
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
