import numpy as np
import pandas as pd


def apply_white_noise(x, A, snr, random_seed):
    '''
    Random sampling from a normal distribution to generate white noise.
    '''
    np.random.seed(random_seed)
    stddev = A.mean() / snr / 4 # approximately noise levels that match snr
    noise = np.random.normal(0, stddev, len(x))
    return x + noise

def apply_pink_noise(x, A, snr, num_sources, random_seed):
    '''
    Stochastic Voss-McCartney algorithm for generating pink noise.

    References:
        https://github.com/AllenDowney/ThinkDSP/blob/master/code/voss.ipynb
        https://www.firstpr.com.au/dsp/pink-noise/
    '''
    np.random.seed(random_seed)
    nrows = len(x)
    ncols = num_sources

    noise = np.full((nrows, ncols), np.nan)
    noise[0, :] = np.random.random(ncols)
    noise[:, 0] = np.random.random(nrows)

    cols = np.random.geometric(0.5, nrows)
    cols[cols >= ncols] = 0
    rows = np.random.randint(nrows, size=nrows)
    noise[rows, cols] = np.random.random(nrows)

    noise = pd.DataFrame(noise)
    noise.fillna(method='ffill', axis=0, inplace=True)
    noise = noise.sum(axis=1).to_numpy()
    noise = (noise - noise.mean())
    noise = (A.mean()/snr) * noise / 2 # approximately noise levels that match snr
    return noise + x

def apply_drift(x, resolution, random_seed):
    np.random.seed(random_seed)
    def sigmoidal(y, m, w, b):
        return 1 / (1 + np.exp( - (y * w + b) )) * m
    y = np.linspace(-1, 1, resolution)
    drift = np.zeros(resolution, dtype='float32')
    n = 10#np.random.randint(0, 11)
    for _ in range(n):
        h = np.random.uniform(-500, 500)  # changed (-1000, 1000)
        w = np.random.uniform(-20, 20)
        b = np.random.uniform(-20, 20)
        drift += sigmoidal(y, h, w, b) / n
    return x + drift
