import numpy as np
import pandas as pd
import multiprocessing
import os

import noise


def compute_derivative(x, n):
    '''
    Computes the n:th order derivative

    The constant multiplied with each derivative is there to increase the
    scale of the gradient for better training of the Autoencoder later on.
    '''
    for _ in range(n):
        x = np.gradient(x) * 10
    return x

def compute_gaussian_peak(x, A, loc, scale, asymmetry=None, eps=1e-7):
    '''
    Computes a normal Gaussian (asymmetry == None)
               or
               modified Gaussian (asymmetry != None)
    '''
    if asymmetry is None:
        return A * np.exp((-(x - loc)**2) / (2 * scale**2))
    return A * np.exp(-1/2*((x - loc)/(eps + scale + asymmetry * (x - loc)))**2)


class Simulator:

    def __init__(self,
                 resolution,
                 num_peaks_range,
                 snr_range,
                 A_range,
                 loc_range,
                 scale_range,
                 asymmetry_range,
                 pink_noise_prob,
                 der_order):

        self.x = np.linspace(0, 1, resolution)
        self.resolution = resolution
        self.num_peaks_range = num_peaks_range
        self.snr_range = snr_range
        self.A_range = A_range
        self.loc_range = loc_range
        self.scale_range = scale_range
        self.asymmetry_range = asymmetry_range
        self.pink_noise_prob = pink_noise_prob
        self.der_order = der_order

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
            chromatogram_noisy = noise.apply_pink_noise(
                chromatogram, self.A, self.snr, 6, random_seed)
        else:
            chromatogram_noisy = noise.apply_white_noise(
                chromatogram, self.A, self.snr, random_seed)

        chromatogram_noisy_drift = noise.apply_drift(
            chromatogram_noisy, self.resolution, random_seed)

        return (
            chromatogram_noisy,
            chromatogram_noisy_drift,
            chromatogram,
            compute_derivative(chromatogram, self.der_order),
            self.loc,
            self.scale
        )



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=190_000)
    parser.add_argument('--der_order', type=int, default=2)
    args = parser.parse_args()

    simulator = Simulator(
        resolution=8192,
        num_peaks_range=(1, 100),
        snr_range=(1.0, 20.0),
        A_range=(25, 250),
        loc_range=(0.05, 0.95),
        scale_range=(0.001, 0.003),
        asymmetry_range=(-0.1, 0.1),
        pink_noise_prob=1.0,
        der_order=args.der_order,
    )

    def save_example(path, random_seed):
        x1, x2, y, y_der, loc, scale = simulator.run(random_seed)
        np.save(path, np.stack([x1, x2, y, y_der]))
        np.save(path+'_prop', np.stack([loc, scale]))

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
    generate('../input/simulations/train', args.N, 90_000)
