#! /usr/bin/env python
# Time-stamp: <2024-09-09 14:39:19 rytis>
# Demonstrates building augmented files, one per each recording.

import numpy as np
from soundfile import read


from rainml.augment import augment
from rainml.encoders import enc, unchain_v1, OvalBox
from rainml.utils import check_bytes

# Some constants
inp_basepath = "./input"
aug_basepath = "/tmp"

target_freq = 44100
num_augment = 10
dsids = [1, 3, 5, 4, 2]

rate_override = 2

def encoder(sig):
    (x, y) = enc(sig, rate=rate_override, interpolation='simple')
    return np.array([x, y]).T

def denoiser(sig, par):
    return unchain_v1(sig, OvalBox(par[0], par[1]))


class AugmentationGenerator:
    ''' Parameter generator for data sampling used in the paper
    It's main characteristics are:
    1. Sampling frequency randomization by +/- 2.5% around the target frequency
    2. Denoiser parameter randomization within provided ranges. The time parameter is uniformly distributed, the amplitude parameter is log-uniformly distributed.
    '''
    def db2amp(t): return pow(10, -0.05*t)
    def amp2db(t): return -20.0*np.log10(t)
    
    def __init__(self, target_freq, lims_x, lims_y, nsamp):
        for t in (lims_x, lims_y):
            assert isinstance(t, (tuple, list, np.array)) and len(t)==2, "limits of the form [min, max] are not provided"
        # Frequencies in the +/- 2.5% range stepped in 100s.
        sampling_range = np.arange(
            100*(int(0.975*target_freq)//100),
            100*(int(1.025*target_freq)//100+1), 100)
        self.nsamp = nsamp
        self.idx = 0
        self.frequencies = np.random.choice(sampling_range, nsamp, replace=True)
        self.rx = np.random.uniform(lims_x[0], lims_x[1], nsamp)
        # We want the amplitude parameters to be log-uniformly distributed
        max_y_db = int(amp2db(lims_y[0]))
        min_y_db = int(amp2db(lims_y[1]))
        self.ry = np.random.choice(db2amp(np.arange(min_y_db, max_y_db)), nsamp, replace=True)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.idx == self.nsamp:
            raise StopIteration
        res = (self.frequencies[self.idx], 0, [self.rx[self.idx], self.ry[self.idx]])
        self.idx += 1
        return res

    def __call__(self):
        return iter(self)



for i, dsid in enumerate([1, 3]):
    inp_wav = f'{inp_basepath}/{dsid}.wav'

    # The quantiles used in the paper are 0.75 for the minimum and 0.99 for the maximum. In retrospect, both boundaries are too extreme, especially the lower one. Something like [0.5, 0.9] would probably have been OK too.
    sig, r = read(inp_wav)
    encoded = encoder(sig)
    qtiles = np.quantile(np.abs(np.diff(encoded, axis=0)), [0.75, 0.99], axis=0)
    lims_x = [max([1.1, qtiles[0, 0]]), qtiles[1, 0]]
    lims_y = [qtiles[0, 1], qtiles[1, 1]]

    # # If we wished to combine all inputs into a single augmented output:
    # gen.append(AugmentationGenerator(target_freq, lims_x, lims_y, num_augment))
    # inp.append(inp_wav)
    inp = inp_wav
    gen = AugmentationGenerator(target_freq, lims_x, lims_y, num_augment)
    aug_path = f'{aug_basepath}/{dsid}.tfrec'
    # The next line would be commented out for a combination case.
    augment([inp], aug_path, encoder, rate_override, denoiser, [gen])
    check_bytes([aug_path])
    print('check:', aug_path)

# Next steps:
# from rainml.augment import load_augmented
# data = load_augmented(['/tmp/1.tfrec'])
# etc.
