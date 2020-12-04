import numpy as np
import scipy.signal


def mean_relative_error(y, x):
    return np.mean(np.absolute(y - x) / (1 + np.absolute(y)))

def mean_absolute_error(y, x):
    return np.mean(np.absolute(y - x))

def mean_squared_error(y, x):
    return np.mean(np.square(y - x))

def root_mean_squared_error(y, x):
    return np.sqrt(mean_squared_error(y, x))

def root_mean_squared_error_peaks(y, x, peak_regions):
    return root_mean_squared_error(y[peak_regions], x[peak_regions])

def peak_signal_to_noise_ratio(y, x, maxval=250):
    mse = mean_squared_error(y, x)
    return 20 * np.log10(maxval / np.sqrt(mse))

def signal_to_noise_ratio(y, x):
    eps=1e-3
    H = x[scipy.signal.find_peaks(y)[0]].mean()
    idx = np.where(y < eps)[0]
    h = x[idx].std()*4
    return H/(h+eps)

def compute_metrics(y, x, peak_regions):
    psnr = peak_signal_to_noise_ratio(y, x)
    snr = signal_to_noise_ratio(y, x)
    rmse = root_mean_squared_error(y, x)
    rmse_p = root_mean_squared_error_peaks(y, x, peak_regions)
    return rmse, rmse_p, psnr, snr

def compute_f1_score(t_peak_regions, loc, scale, idx_detected_peaks):
    mean = loc.copy()
    std = scale.copy()
    TP = FP = FN = 0
    predictions = []
    for i in idx_detected_peaks:
        match = np.where(
            (mean-std < t_peak_regions[i]) & (mean+std > t_peak_regions[i]))[0]
        if len(match) > 0:
            TP += 1
            predictions.append(i)
            mean[match[0]] = -1
        else:
            FP += 1
    FN = len(mean)-TP
    return (TP/(TP+(1/2)*(FP+FN))), predictions
