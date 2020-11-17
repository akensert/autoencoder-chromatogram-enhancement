import scipy.signal
import functools
import multiprocessing
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import signal_processing

def mean_relative_error(y, x):
    return np.mean(np.absolute(y - x) / (1 + np.absolute(y)))

def mean_absolute_error(y, x):
    return np.mean(np.absolute(y - x))

def mean_squared_error(y, x):
    return np.mean(np.square(y - x))

def peak_signal_to_noise_ratio(y, x, maxval=250):
    mse = np.mean(np.square(y-x))
    return 20 * np.log10(maxval / np.sqrt(mse))

def signal_to_noise_ratio(y, x):
    eps=1e-3
    H = x[scipy.signal.find_peaks(y)[0]].mean()
    idx = np.where(y < eps)[0]
    h = x[idx].std()*4
    return H/(h+eps)

def grid_search(f, x, y, g, parameters):
    grid = itertools.product(*parameters)
    best_error = float('inf')
    for p in grid:
        if g is not None:
            x_ = g(x)
            x_ = f(x_, *p)
        else:
            x_ = f(x, *p)
        mre = mean_relative_error(y, x_)
        mae = mean_absolute_error(y, x_)
        mse = mean_squared_error(y, x_)
        psnr = peak_signal_to_noise_ratio(y, x_)
        snr = signal_to_noise_ratio(y, x_)
        if mae < best_error:
            best_error = mae
            best_mre   = mre
            best_mae   = mae
            best_mse   = mse
            best_psnr  = psnr
            best_snr   = snr
            best_param = p
            best_x_    = x_

    return best_mre, best_mae, best_mse, best_psnr, best_snr, best_param



if __name__ == '__main__':
    import argparse
    np.set_printoptions(suppress=True)
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=100)
    parser.add_argument('--T', type=str, default='white')

    args = parser.parse_args()

    cae = tf.saved_model.load('../output/model')

    savgol_lengths = [
        5, 13, 21, 29, 37, 45, 53, 61, 69, 77,
        85, 93, 101, 109, 117, 125, 133, 141, 149, 157
    ]
    savgol_orders =  [2, 3, 4]
    savgol_n_iters = [1, 2, 3]

    gaussian_lengths = [
        5, 13, 21, 29, 37, 45, 53, 61, 69, 77,
        85, 93, 101, 109, 117, 125, 133, 141, 149, 157
    ]
    gaussian_sigmas =  [
        1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
    ]
    gaussian_n_iters = [1, 2, 3]

    als_lam =     [1e6, 1e7, 1e8, 1e9]
    als_p =       [1e-2, 1e-3, 1e-4]
    als_n_iters = [1, 2, 3]

    polynomial_orders =  [2, 6, 12, 24, 48]
    polynomial_n_iters = [1, 2, 4, 8]


    def main(index):

        x1, x2, y = np.load(f'../input/simulations/test_{args.T}/chromatogram_{index}.npy')

        mre1, mae1, mse1, psnr1, snr1, param1 = grid_search(
            f=signal_processing.savgol_filter,
            x=x1,
            y=y,
            g=None,
            parameters=(savgol_lengths, savgol_orders, savgol_n_iters)
        )

        mre2, mae2, mse2, psnr2, snr2, param2 = grid_search(
            f=signal_processing.gaussian_filter,
            x=x1,
            y=y,
            g=None,
            parameters=(gaussian_lengths, gaussian_sigmas, gaussian_n_iters)
        )

        g = functools.partial(
            signal_processing.savgol_filter,
            length=param1[0],
            order=param1[1],
            n_iter=param1[2]
        )

        mre3, mae3, mse3, psnr3, snr3, param3 = grid_search(
            f=signal_processing.als_baseline_correction,
            x=x2,
            y=y,
            g=g,
            parameters=(als_lam, als_p, als_n_iters,)
        )

        mre4, mae4, mse4, psnr4, snr4, param4 = grid_search(
            f=signal_processing.polynomial_baseline_correction,
            x=x2,
            y=y,
            g=g,
            parameters=(polynomial_orders, polynomial_n_iters)
        )

        x_ = cae.smooth(x1.astype(np.float32))
        mre5a = mean_relative_error(y, x_)
        mae5a = mean_absolute_error(y, x_)
        mse5a = mean_squared_error(y, x_)
        psnr5a = peak_signal_to_noise_ratio(y, x_.numpy())
        snr5a = signal_to_noise_ratio(y, x_.numpy())

        x_ = cae.smooth(x2.astype(np.float32))
        mre5b = mean_relative_error(y, x_)
        mae5b = mean_absolute_error(y, x_)
        mse5b = mean_squared_error(y, x_)
        psnr5b = peak_signal_to_noise_ratio(y, x_.numpy())
        snr5b = signal_to_noise_ratio(y, x_.numpy())

        return [
            (mre1, mae1, mse1, psnr1, snr1),
            (mre2, mae2, mse2, psnr2, snr2),
            (mre3, mae3, mse3, psnr3, snr3),
            (mre4, mae4, mse4, psnr4, snr4),
            (mre5a, mae5a, mse5a, psnr5a, snr5a),
            (mre5b, mae5b, mse5b, psnr5b, snr5b),
        ]




    # Could not make multiprocessing work when including the
    # autoencoder (cae.smooth(...))
    # with multiprocessing.Pool(12) as pool:
    #     for c in tqdm.tqdm(pool.imap(main, range(100)), total=20_000):
    #         output.append(c)

    # run without multiprocessing instead
    output=[]
    for i in tqdm.tqdm(range(args.N)):
        output.append(main(i))

    # save results
    np.save(f'../output/results_{args.T}.npy', np.array(output))
    print(f'results ({args.T} noise) =\n', np.array(output).mean(axis=0))
    print('\n')
