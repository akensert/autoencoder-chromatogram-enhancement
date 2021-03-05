import functools
import multiprocessing
import tqdm
import numpy as np
import itertools
import pywt

import signal_processing
import metrics

# Best params obtained from running this script:
# ----
# savgol param         = (45, 4, 3)
# gaussian param       = (13, 8, 1)
# wavelet param (top5) = ('bior6.8', 3), ('coif3', 3), ('sym9', 3)
# als param            = (1000000000.0, 0.0001, 3)
# polynomial param     = (12, 8)
# ---
# These 'best params' are hardcoded in evaluate.py, to be run for evaluation

def run(param, f, g):
    '''
    Smoothens N (e.g 10000) noisy simulated chromatograms
    and computes the mean of the mean squared error.
    '''
    errors = []
    for i in range(0, 100_000, 10):
        x1, x2, y, _ =  np.load(
            '../input/simulations/train/chromatogram_{}.npy'.format(i))
        if g is not None:
            x_ = f(g(x2), *param)
        else:
            x_ = f(x1, *param)
        errors.append(metrics.mean_squared_error(y, x_))
    return (param, np.mean(errors))


# SAVITZKY-GOLAY SMOOTHING HYPERPARAMETER SEARCH
length = [5, 13, 21, 29, 37, 45, 53, 61, 69, 77, 85, 93, 101]
order =  [2, 3, 4]
n_iter = [1, 2, 3, 4]
grid = list(itertools.product(*(length, order, n_iter)))
with multiprocessing.Pool(12) as pool:
    savgol_param = sorted([
        e for e in tqdm.tqdm(
            pool.imap(
                functools.partial(
                    run,
                    f=signal_processing.savgol_filter,
                    g=None
                ),
                grid
            ),
            total=len(grid),
            desc='savgol parameter search'
        )
    ], key=lambda x: x[1])[0][0]

# GAUSSIAN SMOOTHING HYPERPARAMETER SEARCH
length = [5, 13, 21, 29, 37, 45, 53, 61, 69, 77, 85, 93, 101,]
sigma =  [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20]
n_iter = [1, 2, 3]
grid = list(itertools.product(*(length, sigma, n_iter)))
with multiprocessing.Pool(12) as pool:
    gaussian_param = sorted([
        e for e in tqdm.tqdm(
            pool.imap(
                functools.partial(
                    run,
                    f=signal_processing.gaussian_filter,
                    g=None
                ),
                grid
            ),
            total=len(grid),
            desc='gaussian parameter search'
        )
    ], key=lambda x: x[1])[0][0]

# WAVELET SMOOTHING HYPERPARAMETER SEARCH
kind = pywt.wavelist(kind='discrete')
level = [1, 2, 3, 4]
grid = list(itertools.product(*(kind, level)))
with multiprocessing.Pool(12) as pool:
    wavelet_param = sorted([
        e for e in tqdm.tqdm(
            pool.imap(
                functools.partial(
                    run,
                    f=signal_processing.wavelet_filter,
                    g=None
                ),
                grid
            ),
            total=len(grid),
            desc='wavelet parameter search'
        )
    ], key=lambda x: x[1])[:5]

# ALS FITTING HYPERPARAMETER SEARCH
lam = [1e6, 1e7, 1e8, 1e9]
p = [1e-2, 1e-3, 1e-4]
n_iter = [2, 3, 4]
grid = list(itertools.product(*(lam, p, n_iter)))
with multiprocessing.Pool(12) as pool:
    als_param = sorted([
        e for e in tqdm.tqdm(
            pool.imap(
                functools.partial(
                    run,
                    f=signal_processing.als_baseline_correction,
                    g=functools.partial(
                        signal_processing.savgol_filter,
                        length=savgol_param[0],
                        order=savgol_param[1],
                        n_iter=savgol_param[2]
                    )
                ),
                grid
            ),
            total=len(grid),
            desc='als parameter search'
        )
    ], key=lambda x: x[1])[0][0]

# POLYNOMIAL FITTING HYPERPARAMETER SEARCH
order = [4, 6, 8, 10, 12]
n_iter = [2, 4, 6, 8]
grid = list(itertools.product(*(order, n_iter)))
with multiprocessing.Pool(12) as pool:
    polynomial_param = sorted([
        e for e in tqdm.tqdm(
            pool.imap(
                functools.partial(
                    run,
                    f=signal_processing.polynomial_baseline_correction,
                    g=functools.partial(
                        signal_processing.savgol_filter,
                        length=savgol_param[0],
                        order=savgol_param[1],
                        n_iter=savgol_param[2]
                    )
                ),
                grid
            ),
            total=len(grid),
            desc='polynomial parameter search'
        )
    ], key=lambda x: x[1])[0][0]

print('savgol_param         =', savgol_param)
print('gaussian_param       =', gaussian_param)
print('wavelet_param (top5) =', wavelet_param)
print('als_param            =', als_param)
print('polynomial_param     =', polynomial_param)
