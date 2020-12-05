import numpy as np
import scipy.signal
import functools
import multiprocessing
import tqdm

import signal_processing
import metrics


def peak_detection(x, y, t, peak_regions, method1, method2, filter_length=9, height=1, distance=30):
    smooth = scipy.signal.savgol_filter
    y_pred = method1(x)
    if hasattr(method1, '__name__'):
        der_pred = method2(x)
    else:
        der_pred = method2(y_pred)

    idx = scipy.signal.find_peaks(
        -smooth(der_pred[peak_regions], filter_length, 2),
        height=height,
        distance=distance)[0]
    return y_pred, der_pred, idx

def read_data(path):
    x1, x2, y1, _ = np.load(path+'.npy')
    loc, scale = np.load(path+'_prop'+'.npy')
    t = np.linspace(0, 1, 8192)
    return x1, x2, y1, t, loc, scale

def extract_peak_regions(t, loc, scale, dev):
    peak_regions=[]
    for l, s in zip(loc, scale):
        idx = np.where((t>l-dev*s) & (t<l+dev*s))[0]
        peak_regions.extend(idx)
    return list(set(peak_regions))

def deriv(x, n, w, m):
    for i in range(n):
        if i != 0:
            x = np.gradient(scipy.signal.savgol_filter(x, w, 2))*10*m
        else:
            x = np.gradient(x)*10*m
    return x

if __name__ == '__main__':
    import argparse
    np.set_printoptions(suppress=True)
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=100)
    parser.add_argument('--inpath', type=str, default='../output/model')
    parser.add_argument('--T', type=str, default='white')

    args = parser.parse_args()

    cae = tf.saved_model.load(args.inpath)

    def main(index):

        x1, x2, y, t, loc, scale = read_data(f'../input/simulations/test_{args.T}/chromatogram_{index}')
        indices = extract_peak_regions(t, loc, scale, dev=2)
        indices2 = extract_peak_regions(t, loc, scale, dev=3)

        param = (45, 4, 3)
        method1 = functools.partial(signal_processing.savgol_filter, length=param[0], order=param[1], n_iter=param[2])
        method2 = functools.partial(deriv, n=2, w=15, m=1)
        y_pred, der_pred, idx = peak_detection(
            x1, y, t, indices, method1=method1, method2=method2)
        f1_score, matches = metrics.compute_f1_score(t[indices], loc, scale, idx)
        # COMPUTE METRICS
        rmse, rmse_p, psnr, snr = metrics.compute_metrics(y, y_pred, indices2)
        metrics1 = (rmse, rmse_p, psnr, snr, f1_score)

        g = functools.partial(
            signal_processing.savgol_filter,
            length=param[0], order=param[1],n_iter=param[2])

        param = (13, 8, 1)
        method1 = functools.partial(signal_processing.gaussian_filter, length=param[0], sigma=param[1], n_iter=param[2])
        method2 = functools.partial(deriv, n=2, w=15, m=1)
        y_pred, der_pred, idx = peak_detection(
            x1, y, t, indices, method1=method1, method2=method2)
        f1_score, matches = metrics.compute_f1_score(t[indices], loc, scale, idx)
        # COMPUTE METRICS
        rmse, rmse_p, psnr, snr = metrics.compute_metrics(y, y_pred, indices2)
        metrics2 = (rmse, rmse_p, psnr, snr, f1_score)

        param = (1000000000.0, 0.01, 3)
        method1a = g
        method1b = functools.partial(signal_processing.als_baseline_correction, lam=param[0], p=param[1], n_iter=param[2])
        method1 = lambda x: method1a(method1b(x))
        method2 = functools.partial(deriv, n=2, w=15, m=1)
        y_pred, der_pred, idx = peak_detection(
            x2, y, t, indices, method1=method1, method2=method2)
        f1_score, matches = metrics.compute_f1_score(t[indices], loc, scale, idx)
        # COMPUTE METRICS
        rmse, rmse_p, psnr, snr = metrics.compute_metrics(y, y_pred, indices2)
        metrics3 = (rmse, rmse_p, psnr, snr, f1_score)

        param = (12, 8)
        method1a = g
        method1b = functools.partial(signal_processing.polynomial_baseline_correction, order=param[0], n_iter=param[1])
        method1 = lambda x: method1a(method1b(x))
        method2 = functools.partial(deriv, n=2, w=15, m=1)
        y_pred, der_pred, idx = peak_detection(
            x2, y, t, indices, method1=method1, method2=method2)
        f1_score, matches = metrics.compute_f1_score(t[indices], loc, scale, idx)
        # COMPUTE METRICS
        rmse, rmse_p, psnr, snr = metrics.compute_metrics(y, y_pred, indices2)
        metrics4 = (rmse, rmse_p, psnr, snr, f1_score)

        method1 = lambda x: cae.smooth(x).numpy()
        method2 = lambda x: cae.smooth_der(x).numpy()
        y_pred, der_pred, idx = peak_detection(
            x1, y, t, indices, method1=method1, method2=method2)
        f1_score, matches = metrics.compute_f1_score(t[indices], loc, scale, idx)
        # COMPUTE METRICS
        rmse, rmse_p, psnr, snr = metrics.compute_metrics(y, y_pred, indices2)
        metrics5 = (rmse, rmse_p, psnr, snr, f1_score)

        y_pred, der_pred, idx = peak_detection(
            x2, y, t, indices, method1=method1, method2=method2)
        f1_score, matches = metrics.compute_f1_score(t[indices], loc, scale, idx)
        # COMPUTE METRICS
        rmse, rmse_p, psnr, snr = metrics.compute_metrics(y, y_pred, indices2)
        metrics6 = (rmse, rmse_p, psnr, snr, f1_score)

        return [
            metrics1, metrics2, metrics5, metrics3, metrics4, metrics6,
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
    np.save(f'../output/results_{args.T}.npy', np.nan_to_num(np.array(output)))
    print(f'results ({args.T} noise) =\n', np.nanmean(np.array(output), axis=0))
    print('\n')

    with open('output.txt', 'a') as f:
        f.write('\n\n')
        for m in np.nanmean(np.array(output), axis=0):
            for n in m:
                f.write('{:.5f},'.format(n))
            f.write('\n')
