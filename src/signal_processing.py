import numpy as np
import math
import pywt
import scipy.signal
import scipy.linalg
import scipy.sparse


def gaussian_filter(x, length, sigma, n_iter):
    n = np.arange(0, length) - (length - 1.0) / 2
    f = np.exp(-1/2 * (n / sigma)**2)
    f = f / f.sum()
    for _ in range(n_iter):
        x = np.convolve(f, x, 'same')
    return x

def savgol_filter(x, length, order, n_iter):
    for _ in range(n_iter):
        x = scipy.signal.savgol_filter(x, length, order)
    return x

def als_baseline_correction(x, lam, p, n_iter):
    '''
    Asymmetric Least-Squares baseline correction.

    Reference:
        Asymmetric Least Squares Smoothing by P. Eilers and H. Boelens in 2005.

    Notice:
        D = sparse.csc_matrix(np.diff(np.eye(L), 2)) could be used instead.
        However, this dense matrix diff computation could bring about memory issues.
    '''
    L = len(x)
    D = scipy.sparse.diags([1,-2,1],[0,-1,-2], shape=(L, L-2))
    w = np.ones(L)
    for i in range(n_iter):
        W = scipy.sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = scipy.sparse.linalg.spsolve(Z, w*x)
        w = p * (x > z) + (1-p) * (x < z)
    return x - z

def polynomial_baseline_correction(x, order=3, n_iter=100):
    '''
    Polynomial baseline correction.

    Reference:
        https://github.com/lucashn/peakutils/blob/master/peakutils/baseline.py
    '''
    base = x.copy()
    coeffs = np.ones(order)

    cond = math.pow(abs(x).max(), 1. / order)
    y = np.linspace(0., cond, base.size)
    z = x.copy()

    vander = np.vander(y, order)
    vander_pinv = scipy.linalg.pinv2(vander)

    for _ in range(n_iter):
        coeffs = np.dot(vander_pinv, base)
        z = np.dot(vander, coeffs)
        base = np.minimum(base, z)
    return x - z

def wavelet_filter(x, kind='sym9', level=3):
    def madev(d, axis=None):
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)

    coeff = pywt.wavedec(x, kind, mode="per")
    sigma = (1/0.6745) * madev(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, kind, mode='per')

def apply_interpolation(x, y, target_size):
    '''
    Resizes a chromatogram to a given size/resolution
    '''
    f = scipy.interpolate.interp1d(x, y)
    xnew = np.linspace(x.min(), x.max(), target_size)
    ynew = f(xnew)
    return xnew, ynew
