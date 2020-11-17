import numpy as np
import pandas as pd
import multiprocessing
import os
from scipy.stats import exponnorm

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
    y = np.linspace(-1, 1, 8192)
    drift = np.zeros(resolution, dtype='float32')
    n = 10
    for _ in range(n):
        h = np.random.uniform(-500, 500)  # changed (-1000, 1000)
        w = np.random.uniform(-20, 20)
        b = np.random.uniform(-20, 20)
        drift += sigmoidal(y, h, w, b) / n
    return x + drift

def compute_gaussian_peak(x, A, loc, scale, asymmetry=None, epsilon=1e-7):
    '''
    Computes a normal Gaussian (asymmetry == None)
               or
               modified Gaussian (asymmetry != None)
    '''
    if asymmetry is None:
        return A * np.exp((-(x - loc)**2) / (2 * scale**2))
    return A * np.exp(-1/2 * ((x - loc)/(epsilon + scale + asymmetry * (x - loc)))**2)


class Simulator:

    def __init__(self,
                 resolution,
                 num_peaks_range,
                 snr_range,
                 A_range,
                 loc_range,
                 scale_range,
                 asymmetry_range,
                 pink_noise_prob):

        self.x = np.linspace(0, 1, resolution)
        self.resolution = resolution
        self.num_peaks_range = num_peaks_range
        self.snr_range = snr_range
        self.A_range = A_range
        self.loc_range = loc_range
        self.scale_range = scale_range
        self.asymmetry_range = asymmetry_range
        self.pink_noise_prob = pink_noise_prob

    def run(self, random_seed):

        np.random.seed(random_seed)

        self.num_peaks = np.random.randint(*self.num_peaks_range)
        self.A = np.random.uniform(*self.A_range, size=(self.num_peaks,))
        self.loc = np.random.uniform(*self.loc_range, size=(self.num_peaks,))
        self.scale = np.random.uniform(*self.scale_range, size=(self.num_peaks,))
        if self.asymmetry_range is not None:
            self.asymmetry = np.random.uniform(*self.asymmetry_range, size=(self.num_peaks,))
        else:
            self.asymmetry = [None]*self.num_peaks

        chromatogram = np.zeros_like(self.x)
        for a, m, s, t in zip(self.A, self.loc, self.scale, self.asymmetry):
            chromatogram += compute_gaussian_peak(self.x, a, m, s, t)

        self.snr = np.random.uniform(*self.snr_range)


        if self.pink_noise_prob > np.random.random():
            chromatogram_noisy = apply_pink_noise(chromatogram, self.A, self.snr, 10, random_seed)
        else:
            chromatogram_noisy = apply_white_noise(chromatogram, self.A, self.snr, random_seed)

        chromatogram_noisy_drift = apply_drift(chromatogram_noisy, self.resolution, random_seed)

        return chromatogram_noisy, chromatogram_noisy_drift, chromatogram



if __name__ == '__main__':

    simulator = Simulator(
        resolution=8192,
        num_peaks_range=(1, 100),
        snr_range=(1.0, 20.0),
        A_range=(25, 250),
        loc_range=(0.05, 0.95),
        scale_range=(0.001, 0.003),
        asymmetry_range=(-0.1, 0.1),
        pink_noise_prob=1.0,
    )

    def save_example(path, random_seed):
        x, y, z = simulator.run(random_seed)
        np.save(path, np.stack([x, y, z]))

    def generate(path, n, seed):
        if not(os.path.isdir(path)): os.makedirs(path)
        paths = [path + '/' + 'chromatogram_{}'.format(i) for i in range(n)]
        with multiprocessing.Pool() as pool:
            for c in pool.starmap(
                save_example, zip(paths, range(seed, n+seed))): pass



    simulator.pink_noise_prob = 1

    generate('../input/simulations/test_pink', 10_000, 80_000)

    simulator.pink_noise_prob = 0

    generate('../input/simulations/test_white',  10_000, 80_000)

    simulator.pink_noise_prob = 0.5

    generate('../input/simulations/train', 190_000, 90_000)
