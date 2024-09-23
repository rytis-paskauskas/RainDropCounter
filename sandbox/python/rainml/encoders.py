#! /usr/bin/env python
# Time-stamp: <2024-08-19 11:32:55 rytis>
#
# Clone of audio_minimax.py
# To be replaced.
# This is a python module for minimax mapping of audio signals,
# based on audio_20220709_minimax_test.py

import numpy as np
from scipy.signal import argrelextrema
from soundfile import read

def _simple_correction(x_ref, y_ref, yn, yp):
    d1 = 0.5*(yp-yn)
    d2 = 2.0*y_ref-yp-yn
    return (x_ref + (d1/d2 if d2 != 0 else x_ref),
            y_ref + (0.5*d1*d1/d2 if d2 != 0 else y_ref))

def _bracket(sig_y, interpolation=None):
    """
    Identifies local minima and maxima of a signal represented by sig_y(i)
    with or without interpolation. It is assumed that the 'x' coordinate is integer spaced: 0,1,...
    Input
    - sig_y : the signal sampled at fixed rate (for this routine the rate is 1)
    Output
    - ((max_x,min_x), (max_y, min_y))
    Constraints
    - all output arrays are of the same length
    - max_x, min_x: locations (the 'x' coordinates) of, respectively, maximum and minimum amplitudes.
    - max_y, min_y: the values of, respectively, the maximum and minimum amplitudes
    - minima and maxima are interlaced
    - maximum comes first: max_x[i]<min_x[i] for all i.

    
    interpolation: optional interpolation parameter. Currently, the supported
    options are:
    - None (default) : no interpolation.
    - 'simple' : quadratic interpolation based on three points.

    Returns (max_x,max_y) and (min_x, min_y) which are sequences of
    coordinate and value pairs of, respectively, local maxima and
    minima.

    If interpolation='none' the minima and maxima are guaranteed to be interlaced:
    each pair of consequtive
    minima brackets a maximum, and vice versa:
    max_x[i] < min_x[i] < max_x[i+1] and
    min_x[i] < max_x[i+1] < min_x[i+1]

    When interpolation='simple' the interlacing is no longer guaranteed (but it still prevails)
    That is because two consequtive quadratic interpolations do not guarantee this feature.
    Example: sig=[0.241089, 0.286865, 0.283813, 0.332642]
    This sequence has a maximum at sig[1] and a minimum at sig [2].
    The quadratic interpolation yields the respective corrections as
    max: x*=1.875, y*=0.291538
    min: x*=1.117, y*=0.278765
    The max raced forward, and the min was pushed backward, resulting in switched positions.

    The relative extrema are arrange so that the first point is always
    a maximum: max_x[0] < min_x[0] (guaranteed if interpolation='none')

    Care is taken to eliminate 'the flats'.
    """
    sig_x = np.arange(len(sig_y))
    assert len(sig_x) == len(sig_y)
    # Eliminate the flats
    flats = np.where(sig_y[:-1] == sig_y[1:])[0]
    # flats = []
    # for i in range(1, len(sig_y)):
    #     if sig_y[i] == sig_y[i-1]:
    #         flats.append(i)
    x = np.array(np.delete(sig_x, flats), dtype=float)
    y = np.delete(sig_y, flats).astype('float')
    # arrays holding (x,y) for maxima and minima, respectively, such that
    # each  minimum is between two maxima and vv.
    idx = argrelextrema(y, np.greater)[0]
    if not np.any(idx):
        max_x = np.array([], dtype=float)
        max_y = np.array([], dtype=float)
        # max_a = np.array([], dtype=float)
    else:
        # max_a = 2.0*y[idx] - y[idx-1] - y[idx+1]
        if (interpolation is None) or (interpolation == 'none'):
            max_x = x[idx]
            max_y = y[idx]
        elif interpolation == 'simple':
            max_x, max_y = list(np.asarray(list(map(_simple_correction,x[idx],y[idx],y[idx-1],y[idx+1]))).T)

    idx = argrelextrema(y, np.less)[0]
    if np.any(idx):
        # min_a = 2.0*y[idx] - y[idx-1] - y[idx+1]
        if (interpolation is None) or (interpolation == 'none'):
            min_x = x[idx]
            min_y = y[idx]
        elif interpolation == 'simple':
            # With interpolation
            min_x, min_y = list(np.asarray(list(map(_simple_correction,x[idx],y[idx],y[idx-1],y[idx+1]))).T)
    if np.any(min_x) and np.any(max_x):
        i0=0 if max_x[0] < min_x[0] else 1
        mx = np.zeros(len(min_x)+len(max_x), dtype=float)
        my = np.zeros(len(mx), dtype=float)
        mx[i0::2] = max_x
        my[i0::2] = max_y
        #ma[i0::2] = max_a
        mx[1-i0::2] = min_x
        my[1-i0::2] = min_y
        #ma[1-i0::2] = min_a
    else:
        mx = []
        my = []
    # dealing with interpolation issues:
    # swap time and amplitude values where
    # this is a hackish hack...
    if interpolation == 'simple':
        for i in np.where(mx[:-1]>mx[1:])[0]:
            mx[i], mx[i+1] = mx[i+1], mx[i]
            # my[i], my[i+1] = my[i+1], my[i]

    return mx, my


def enc(sig, **kwargs):
    """ Encoder v.2. (2022-10-11)
    Encodes audio signal. Optional parameters:
    :param rate: encode for a specific sampling rate (default is 1 I think...)
    :param interpolation: options are 'none' (default) or 'simple' (quadratic).
    :return: (t, a) where `t' is time and `a' is amplitude
    """
    interp = kwargs.pop('interpolation', None)
    if interp == 'none':
        interp = None
    x, y = _bracket(sig, interp) # x, y, a = 
    # Scale if needed
    if 'rate' in kwargs:
        nrm = 1.0/float(kwargs.pop('rate'))
        x = nrm*x
    return ([] if not np.any(x) else x,
            [] if not np.any(y) else y# ,
            # [] if not np.any(a) else a
            )


def unchain(encoded, inlier_function):
    """ Denoise using the unchaining algorithm based on a discriminator function inlier_function.
    Inlier function is a function that acts on the differential encoded data ( da, dt ) and returns True if the datum lies within a cluster. See quadraticFormInlier for an example.

    encoded: input data, shape=[N,2], generated from enc2 """

    tmp = np.diff(encoded, 1, axis=0)
    tmp[:, 1] = np.abs(tmp[:, 1])
    a = inlier_function(tmp)
    _a = np.diff(a)
    b = np.append(_a[0], _a)
    _, c = np.unique(b.cumsum(), return_index=True)
    # Inclusive ranges:
    out = [(t[1], t[2]) for t in zip(a[c], c, np.append(c[1:], len(tmp))) if t[0]]
    elim = np.array([], dtype='uint8')
    for elt in out:
        s = elt[0]
        e = elt[1] + 1
        n = e - s
        nr = (n % 2)
        if nr == 0:             # Even sequence: elimiate all
            elim = np.append(elim, np.arange(s, e))
        else:                   # Odd sequence: leave dominant
            pos = 0
            if encoded[s, 1] > encoded[s+1, 1]:     # maximum-dominated
                pos = s + encoded[s:e, 1].argmax()
            else:               # minimum-dominated
                pos = s + encoded[s:e, 1].argmin()
            elim = np.append(elim, np.arange(s, pos))
            elim = np.append(elim, np.arange(pos+1, e))
    # return np.delete(encoded, elim, axis=0), out, elim
    return np.delete(encoded, elim, axis=0)

def unchain_v1(encoded, inlier_function):
    """ RainML denoiser with using the unchaining algorithm with even/odd chain distinction. As per our usage, the discriminator inlier_function < 1 implies noise.  
    Inlier function is a function that acts on the differential encoded data ( da, dt ) and returns True if the datum lies within a cluster. See quadraticFormInlier for an example.

    encoded: input data, shape=[N,2], generated from enc2 """

    tmp = np.diff(encoded, 1, axis=0)
    tmp[:, 1] = np.abs(tmp[:, 1])
    a = inlier_function(tmp)
    _a = np.diff(a)
    b = np.append(_a[0], _a)
    _, c = np.unique(b.cumsum(), return_index=True)
    # Inclusive ranges:
    out = [(t[1], t[2]) for t in zip(a[c], c, np.append(c[1:], len(tmp))) if t[0]]
    elim = np.array([], dtype='uint8')
    for elt in out:
        s = elt[0]
        e = elt[1] + 1
        n = e - s
        nr = (n % 2)
        if nr == 0:             # Even sequence: elimiate all
            elim = np.append(elim, np.arange(s, e))
        else:                   # Odd sequence: leave dominant
            pos = 0
            if encoded[s, 1] > encoded[s+1, 1]:     # maximum-dominated
                pos = s + encoded[s:e, 1].argmax()
            else:               # minimum-dominated
                pos = s + encoded[s:e, 1].argmin()
            elim = np.append(elim, np.arange(s, pos))
            elim = np.append(elim, np.arange(pos+1, e))
    # return np.delete(encoded, elim, axis=0), out, elim
    return np.delete(encoded, elim, axis=0)
#
# Various inliers; we'll use OvalBox

class Rectangle:
    """ rectangle(lx, ly) - defines a rectangular box: 0<=x<=lx, 0<=y<=ly. """

    def __init__(self, lx, ly):
        assert (lx >= 0) and (
            ly >= 0), "incorrect parameters"
        self.lx = lx
        self.ly = ly

    def __call__(self, d):
        # for each element of the vector checks that x<=lx and y<=ly
        if type(d) == list:
            d = np.array(d)
        assert (len(d.shape) == 2) and (
            d.shape[1] >= 2), "bad input dimensions"
        return np.logical_and(d[:,0]<=self.lx, d[:,1]<=self.ly)

class OvalBox:
    """ oval_box(lx, ly) - defines an oval box with radii: lx, ly. Implementation can be varied, e.g. ellipsis, quartic , etc. """

    def _oval_func_(self, v):
        return v[0]*v[0]*v[0] + v[1]*v[1]*v[1] <= 1.0

    def __init__(self, lx, ly):
        assert (lx > 0) and (
            ly > 0.0), "incorrect parameters"
        self.norm = [lx, ly]
        self.lx = lx
        self.ly = ly

    def __call__(self, d):
        # for each element of the vector checks that x, y are inside the oval box
        if type(d) == list:
            d = np.array(d)
        assert (len(d.shape) == 2) and (
            d.shape[1] >= 2), "bad input dimensions"
        return np.array([self._oval_func_(d[i, ...]/self.norm) for i in range(d.shape[0])])

class Ellipsis:
    """ outlier_fraction(lambda) - defines the radius of the elipsoid as
    lambda: in [0, 1) = Prob ( outlier is noise)
    lambda determines the radius, scaling of covariance matrix.
    Formula: radius(lambda) = sqrt(-2 ln(1-lambda))
    Lmabda=1 is not allowed, as it implies that no data is returned.
    Examples:
    radius(0) = 0
    radius(0.1) = 0.459
    radius(0.5) = 1.177
    radius(0.7) = 1.665
    radius(0.9) = 2.145
    radius(0.99) = 3.035
    """

    def __init__(self, w, b, noise_probability):
        assert (noise_probability >= 0) and (
            noise_probability <= 1.0), "incorrect noise probability"
        assert type(w) in [list, np.ndarray]
        assert type(b) in [list, np.ndarray]
        if type(w) == list:
            w = np.array(w)
        if type(b) == list:
            b = np.array(b)
        assert w.shape == (2, 2), "weight not a 2x2 matrix"
        assert b.shape == (2,), "bias not a 2x1 vector"
        self.w = w
        self.b = b
        self.threshold = np.sqrt(-2.0*np.log(1.0 - noise_probability)) if noise_probability<1 else np.inf
        print('radius = ', self.threshold)

    def __call__(self, encoded):
        # returns a vector of (x[i]-bias) * w * (x[i]-bias)
        # matrix vector multiplication is done against the last dimension of x.
        assert type(encoded) in [list, np.ndarray], "wrong input type"
        if type(encoded) == list:
            encoded = np.array(encoded)
        assert (len(encoded.shape) == 2) and (
            encoded.shape[1] >= 2), "bad input dimensions"
        # return np.array([(encoded.T[..., i]-self.b)@self.w@(encoded.T[..., i]-self.b) for i in range(encoded.shape[0])]) < self.threshold
        if self.threshold is np.inf:
            return encoded.shape[0]*[True]
        return np.array([(encoded[i, ...]-self.b)@self.w@(encoded[i, ...]-self.b) for i in range(encoded.shape[0])]) < self.threshold

def enc_to_cov(enc, n):
    """
    Input:
      enc: encoded variable (t,a) (time and amplitude)
      n: number of points to randomly sample.
    Returns: coefficients w, m that normalizes a two-dimensional gaussian rv:
    What it does (more detail): First, we use transformed variable (Δt, |Δa|), then compute it's 2d Cov and Mean.
    Then solve the eigenproblem for the covariance matrix to get Λ, E, the eigenvalues and eigenvectors
    Then, the coefficients are computed as w = E (√1/Λ) Eᵀ and mu=mean(enc).
    These coefficients have the property that a variable y=w*(x-mu) is distributed as a standard normal distribution in 2d.
    """
    # d selects the data on which to compute the coefficients.
    d = np.abs(np.diff(enc, axis=0))
    if n < d.shape[0]:
        d = d[np.random.choice(range(d.shape[0]), n, replace=False)]

    mu = np.mean(d, axis=0)
    co = np.cov(d.T)
    ei = np.linalg.eig(co)
    # Here we compute the √(1/Λ)
    L = np.diag(1.0/np.sqrt(ei.eigenvalues))
    w = ei.eigenvectors @ L @ ei.eigenvectors.T
    return (w, mu)




def unchain_new(encoded, noise):
    """ Reworking the unchain1.py algo.
    Here we assume that coordinate is cumulative sum.
    Here's another observation: since we're not altering the data, it is enuogh to return the index.
    2024-03-03 version.
    """

    def unchain1_idx(encoded, noise):
        idx = []
        rx = []
        ry = []
        cn = 0
        nxt = 0
        ckpt_x = 0
        ckpt_y = 0
        ckpt_d = 0
        ckpt_i = 0

        tmp = np.abs(np.diff(encoded, 1, axis=0))
        a = noise(tmp)
        for i in range(1, encoded.shape[0]):
            # print('Pt[', i-1, ']', end=' : ')
            if a[i-1]:  # noise
                # print('Yes noise branch', end=' ≻')
                if cn == 0:         # begin chain
                    # print(" C2 (chain start)")
                    cn = 2
                    # first chain point max or min? sig=1 if max, sig=-1 if min
                    sig = 1 if encoded[i-1, 1] > encoded[i, 1] else -1
                    # # print('first trend:', 'max' if sig == 1 else 'min')
                    nxt = i + 1     # next check for local extremum
                    # Create/update a checkpoint:
                    ckpt_y = encoded[i-1, 1]
                    ckpt_x = encoded[i-1, 0]
                    ckpt_d = 0
                    ckpt_i = i-1
                else:               # continue chain
                    # print("- nxt test", end=' ≻')
                    cn += 1
                    if i - 1 == nxt:
                        # print("- trend test", end=' ≻')
                        if sig*encoded[nxt, 1] > sig*ckpt_y:  # same trend.. 
                            # print(' C5 (move ckpt)')
                            # Move the checkpoint
                            ckpt_x = encoded[i-1, 0]
                            ckpt_y = encoded[i-1, 1]
                            ckpt_i = i-1
                            ckpt_d = 0
                            nxt += 2
                        else:       # break the trend (from min to max and vv.)
                            # print(' C7 (add data + new ckpt)')  # new checkpoint
                            rx.append(ckpt_x)
                            ry.append(ckpt_y)
                            idx.append(ckpt_i)
                            ckpt_x = encoded[i-2, 0]
                            ckpt_y = encoded[i-2, 1]
                            ckpt_i = i-2
                            ckpt_d = encoded[i-1, 0]
                            # then, flip the sign, and reset search
                            sig = - sig
                            nxt += 1
                    else:
                        # print(' C4 (next on chain)')
                        ckpt_d += encoded[i-1, 0]
            else:                   # 'not noise'
                # print('Not noise branch', end=' ≻')
                if cn > 0:          # end of chain - handle it
                    cn = 0
                    # print('- nxt check', end=' ≻')
                    # Testing the last one
                    if i - 1 == nxt:
                        # print('- trend test', end=' ≻')
                        if sig*encoded[nxt, 1] > sig*ckpt_y:  # trend continues. this is the last point though...
                            # print(" C6 (add data + move ckpt)")
                            # Move the checkpoint
                            ckpt_y = encoded[i-1, 1]
                            ckpt_x = encoded[i-1, 0]
                            ckpt_i = i-1
                            ry.append(ckpt_y)
                            rx.append(ckpt_x)
                            idx.append(ckpt_i)
                            ckpt_d = 0
                            ckpt_x = 0
                        else:
                            # print(' C8 (add data + new ckpt)')
                            ry.append(ckpt_y)
                            rx.append(ckpt_x)
                            idx.append(ckpt_i)
                            ckpt_x = encoded[i-2, 0]
                            ckpt_y = encoded[i-2, 1]
                            ckpt_i = i-2
                            ckpt_d = encoded[i-1, 0]
                    else:
                        # print(' C3 (next on chain)')
                        ckpt_d += encoded[i-1, 0]
                else:
                    # print(' C1 (add data + reset ckpt)')  # the simplest block
                    ry.append(encoded[i-1, 1])
                    rx.append(encoded[i-1, 0])
                    idx.append(i-1)
                    ckpt_d = 0
                    ckpt_x = 0
        return idx
    
    return encoded[unchain1_idx(encoded, noise), :]
