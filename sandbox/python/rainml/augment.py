#! /usr/bin/env python
# Time-stamp: <2024-08-19 13:34:23 rytis>
# Author: Rytis
# Augmentation related utilities - Part of Audio Encoder.
# Dataset related utilities (requiring TensorFlow): augment, to_frames, etc.
# 
# Audio Encoder based on Covariance Outlier Chains (Audio ECOC)
# Audio ERCU Dataset:
# Audio Encoder with Randomly sampled Covariance-based Unchaining Dataset
# ------------------------------------------------------------
# Value Proposition:
# This is an intermediate data from which to construct frames.
# Saves labor/computation.
# Having this data, the only remaining thing to do is to slice the data.
# No more resampling, encoding, unchaining/denoising.
# We'll provide tools to make easy workflows.
# ------------------------------------------------------------
# Proposed organization of intermediate TFRecords:
# ------------------------------------------------------------
# For event data: one record per (augmented) recording
# ------------------------------------------------------------
# For other data: can collect multiple recordings under one record, or keep one recording per record, doesn't matter much.
# It is a good practice is to collect audio recordings of the same length (e.g. all audio of 5 seconds, etc)
# Can additionally split by, e.g. ESC50 label, fold, but that's not essential.

#
# Each record can handle
# - One file or a file list.
# - A list of sampling rates.
# - list of denoising threshold probabilities (between 0 and 1, but most useful in the range 0.01 - 0.1)

# The record holds the following fields:
# - name of the original audio recording file
# - the sampling rate with which this recording was done

# -----------------------------------------
# Augment through parameters of:
#  .1 resampling (provide sampling frequency list)
#  .2 preamp / gain
#  .3 unchaining
# 2. In each case, make a tfrecord consisting of:
#  .1 recording file name (original)
#  .2 recording sampling_rate (original)
#  .3 speed (sampling_rate override)
#  --- augmentation ---
#  .4 random preamp/gain (-2, -1, 0)
#  .5 random unchain_matrix (1/2 covariance)
#  .6 random unchain_vector (bias)
#  .7 random unchain_threshold
#  --- data ---
#  .7 encoded_length
#  .8 encoded_data
# Audio Encoding with Randomly sampled covariance nchaining 
# Encoder with Randomized Covariance and Threshold Unchaining
# Audio Encoder based on unchaining and covariance thresholding 

# Audio ERCU datasets
# Audio Encoding with Randomly sampled Covariance Unchaining

import numpy as np
import tensorflow as tf
from soundfile import read
from os import system, remove
from os.path import dirname, isdir, isfile, split
from sys import exc_info
# from rainml_git.encoders import enc, unchain_new, OvalBox
from tempfile import NamedTemporaryFile
import multiprocessing as mp

# DEFAULT_SAMPLE_FRACTION = 0  # 10% is a reasonable default value
# DEFAULT_SAMPLE_MINPTS = 0  # 50 points is a reasonable default value
# DEFAULT_RATE_OVERRIDE = 2   # 
# DEFAULT_BOX_PARAM_Y = 0.5
# DEFAULT_BOX_PARAM_X2RO = 0.6  # Multiplier to rate_override

# def augment(input_paths: list, output_path: str, A: int, **kwargs):
#     """
#     Create an augmented dataset from the original recordings and various randomization options. This dataset is stored in the form of a TFRecord.
#     Its main features is the 'encoded_data' which is byte-encoded float16 'Poincare' encoded time differences and amplitude array of shape [N,2] and
#     'time' which is a byte encoded float64 array corresponding to times of the encoded data. Shape: [N].

#     The minimum parameter set is input_paths and A, which may be set to A=0. In this case a single record will be produced consisting of the encoded original recording.


#     Input:
#     input_paths: mandatory list of source files
#     output_path: mandatory output path
#     A: mandatory number of augmented records for each input (A=0 : no augmentation)
#     Optional input:
#     rate_override: change input rate to this rate (applies to all sampling frequencies. Default: 2
#     sampling_range: list of frequencies to use for resampling (only used if A>0) Defaults to original sampling rate, i.e. no resampling.
#     gain_range: list of gain (preamp) in decibels to used for resampling (only used if A>0). Defaults to 0 i.e. no gain change. You would normally use negative values  (attenuation) or zero (no change).
#     parameter_range: [ [1.2, 1.0], ...]
#     sample_fraction: a fraction of all data to use in the calculation of covariance. Defaults to 0.01.
#     sample_minpts: overrides unchain_take_fraction for small data. This is the minimum points. If the sample does not has this number of points, then disregard. Defaults to 50.


#     -- Sampling method:
#     1. Request A augmented data.
#     If A = 0 we will use 1

#     Then, 1+A parameter combinations will be generated for each input file (+1 for the original recording).
#     So for N files you get N. (1+A) records. Default A=0, so that each file gets one record for the original recording.
#     Default unchaining parameters:

#     Input: various randomization options
#     --- Preamp and sampling range are sampled together.
#     preamp_range: (aka [-2, -1, 0] or None) random preamp (in dB) is applied. Default = None, will use the original preamp.
#     Standard choice for sampling 'around' the main frequency is to give something like np.arange(int(0.95*srate0), int(1.05*srate0)+1)
#     sampling_range : (list or None) random resampling is applied
#     ---
#     Unchaining parameters
#     unch_p_range: unchaining probability parameter (aka np.logspace(-2, -1, 20))
    
#     """

def collect_features_helper(path, rate1, rate2, preamp, param, encoded):
    feature = dict({})
    feature['source_file'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[bytes(path, 'utf-8')]))
    feature['from_rate'] = tf.train.Feature(
        float_list=tf.train.FloatList(value=[float(rate1)]))
    feature['gain'] = tf.train.Feature(
        float_list=tf.train.FloatList(value=[float(preamp)]))
    tmp = tf.io.serialize_tensor(tf.constant(
        param, dtype='float64', name='parameter'), 'float64')
    feature['parameter'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tmp.numpy()]))
    tmp = (float(rate2)/float(rate1))*encoded[:, 0]
    tmp = tf.io.serialize_tensor(tf.constant(
        tmp, dtype='float64', name='time'), 'float64')
    # For convenience the original time sequence is stored separately as 'time' with a high resolution.
    feature['time'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tmp.numpy()]))
    feature['encoded_rate'] = tf.train.Feature(
        float_list=tf.train.FloatList(value=[float(rate2)]))
    tmp = encoded
    # tmp contains encoded time and amplitude, but we make a further step and differentiate time.
    # This avoids overflows and accuracy loss for long sequences. 
    # Final encoded data contains  (delta t, a) in half-float precision
    tmp[0, 0] = 0
    tmp[1:, 0] = tmp[1:, 0] - tmp[:-1, 0]    
    tmp = tf.io.serialize_tensor(tf.constant(
        tmp, dtype='float16', name='encoded'), 'float16')
    feature['encoded_data'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tmp.numpy()]))
    return feature

def hard_worker(encoder, rate_override, denoiser, from_path, sr_par, g_par, unc_par, q):
    """Does the hard work with one set of encoding and denoising parameters
    do_resample: bool
    replacing with sr_par = None => do_resample = False else do_resample=True
    """

    resample_sentence = f"rate -v -s {sr_par}" if (sr_par is not None) else ""
    gain_sentence = f"gain {g_par:.2f}" if (g_par != 0) else ""
    with NamedTemporaryFile() as fp:
        fp.close()
        tmp_path = fp.name + ".wav"
        print(f"[LOG] full_sequence: sox {from_path} -b16 -c1 {tmp_path} {gain_sentence} {resample_sentence} dither -a")
        # print("2")
        rc = system(
            f"sox {from_path} -b16 -c1 {tmp_path} {gain_sentence} {resample_sentence} dither -a"
        )
        assert isfile(tmp_path) and (rc == 0), "sox resampling failed"
        sig, now_rate = read(tmp_path)
        if sr_par is not None:
            assert now_rate == sr_par, f"issue with reading resampled file: expected: {sr_par} found: {now_rate}"
        else:
            sr_par = now_rate
        encoded = encoder(sig)
        if denoiser is not None:
            encoded = denoiser(encoded, unc_par)
        else:
            unc_par = np.array([0, 0], dtype=float)

        # Adding nan, inf test:
        if np.any(np.isnan(encoded)) or np.any(np.isinf(encoded)):
            print("[LOG] augment/hard_worker: nan or inf values found. skipping")
        else:
            feature = collect_features_helper(
                from_path, sr_par, rate_override, g_par, unc_par, encoded)
            example = tf.train.Example(
                features=tf.train.Features(feature=feature))
            serialized = example.SerializeToString()
            q.put(serialized)
            # I have done my work of serializing data
        remove(tmp_path)


def easy_writer(path, q):
    """ The file-meister: Writes serialized data to the file.

    Put None to the queue to signal that we're done.
    """

    writer = tf.io.TFRecordWriter(path)
    counter = 0
    while True:
        serialized = q.get()
        if serialized is None:
            break
        else:
            counter += 1
            writer.write(serialized)
            writer.flush()
    writer.close()
    print("[LOG] easy_writer: wrote", counter, "records.")


def load_augmented(input_paths, **kwargs):
    """ Parse a record into a list of dictionaries.
    Watch out, could be large!
    """
    result = []
    assert type(input_paths) in [list, np.str_]
    for fn in input_paths:
        if not isfile(fn):
            return result
    ds = tf.data.TFRecordDataset(input_paths)
    for raw in ds:
        result.append(parse_single_augmented(raw, **kwargs))
    return result

def parse_single_augmented(raw, **kwargs):
    """ Parse a single record from augmented records.
    You may pass a list of fields to be parsed in the fields=[...] argument.
    """
    e = tf.train.Example()
    e.ParseFromString(raw.numpy())
    result = dict({})
    field_list = kwargs.get('fields', ['source_file', 'from_rate', 'gain', 'parameter', 'time', 'encoded_rate', 'encoded_data'])
    if field_list is None:
        return result
    t = 'source_file'
    if t in field_list:
        result[t] = getattr(getattr(e.features.feature.get(
            t), 'bytes_list'), 'value')[0].decode('utf-8')
    t = 'from_rate'
    if t in field_list:
        result[t] = getattr(getattr(e.features.feature.get(
            t), 'float_list'), 'value')[0]
    t = 'encoded_rate'
    if t in field_list:
        result[t] = getattr(getattr(e.features.feature.get(
            t), 'float_list'), 'value')[0]
    t = 'gain'
    if t in field_list:
        result[t] = getattr(getattr(e.features.feature.get(
            t), 'float_list'), 'value')[0]
    t = 'encoded_data'
    if t in field_list:
        tmp = getattr(getattr(e.features.feature.get(
            t), 'bytes_list'), 'value')[0]
        result[t] =tf.io.parse_tensor(tmp, 'float16').numpy()
    for t in ['time', 'parameter']:
        if t in field_list:
            tmp = getattr(getattr(e.features.feature.get(
                t), 'bytes_list'), 'value')[0]
            result[t] = tf.io.parse_tensor(tmp, 'float64').numpy()
    return result


# Note here the denoiser will take a generic parameter, so it can't be the unchain, or similar.
def augment(input_paths: list, output_path: str, encoder: object, rate_override: int, denoiser=None, generators=None, **kwargs):
    """
    Create an augmented dataset from the original recordings and various randomization options. This dataset is stored in the form of a TFRecord.
    Its main features is the 'encoded_data' which is byte-encoded float16 'Poincare' encoded time differences and amplitude array of shape [N,2] and
    'time' which is a byte encoded float64 array corresponding to times of the encoded data. Shape: [N].

    The minimum parameter set is input_paths and A, which may be set to A=0. In this case a single record will be produced consisting of the encoded original recording.


    Input:
    input_paths: mandatory list of source files
    output_path: mandatory output path
    A: mandatory number of augmented records for each input (A=0 : no augmentation)
    Optional input:
    rate_override: change input rate to this rate (applies to all sampling frequencies. Default: 2
    sampling_range: list of frequencies to use for resampling (only used if A>0) Defaults to original sampling rate, i.e. no resampling.
    gain_range: list of gain (preamp) in decibels to used for resampling (only used if A>0). Defaults to 0 i.e. no gain change. You would normally use negative values  (attenuation) or zero (no change).
    parameter_range: [ [1.2, 1.0], ...]
    sample_fraction: a fraction of all data to use in the calculation of covariance. Defaults to 0.01.
    sample_minpts: overrides unchain_take_fraction for small data. This is the minimum points. If the sample does not has this number of points, then disregard. Defaults to 50.


    -- Sampling method:
    1. Request A augmented data.
    If A = 0 we will use 1

    Then, 1+A parameter combinations will be generated for each input file (+1 for the original recording).
    So for N files you get N. (1+A) records. Default A=0, so that each file gets one record for the original recording.
    Default unchaining parameters:

    Input: various randomization options
    --- Preamp and sampling range are sampled together.
    preamp_range: (aka [-2, -1, 0] or None) random preamp (in dB) is applied. Default = None, will use the original preamp.
    Standard choice for sampling 'around' the main frequency is to give something like np.arange(int(0.95*srate0), int(1.05*srate0)+1)
    sampling_range : (list or None) random resampling is applied
    ---
    Unchaining parameters
    unch_p_range: unchaining probability parameter (aka np.logspace(-2, -1, 20))

    ---
    generator must return: (sampling_value, override_rate, gain_value, param_value)
    
    """


    # tmp_path = "/tmp/augment.wav"

    # input/output paths basic sanity checks
    assert all([isfile(t) for t in input_paths]), "some input are not files"
    assert isdir(dirname(output_path)), "output path not a folder"

    # Queue and its manager
    # Many threads are spawned to speed up the [sox] -- [encode] -- [denoise] workflow
    # One thread is used for the [writer] loop
    try:
        manager = mp.Manager()
        q = manager.Queue()
        pool = mp.Pool(mp.cpu_count()+1)
        jobs = []
        _ = pool.apply_async(easy_writer, (output_path, q,))
        if generators is not None:
            assert type(generators)==list, "not a list: Generators should be a list of generators, one for each input file"
            assert len(generators) == len(input_paths), "not equal length: Generators should be a list of generators, one for each input file"
        for idx, this_path in enumerate(input_paths):
            if generators is None:
                job = pool.apply_async(
                    hard_worker, (encoder, rate_override, denoiser, this_path, None, 0, None, q,)
                )
                jobs.append(job)
            else:
                for sr, g, p in generators[idx]():
                    job = pool.apply_async(
                        hard_worker, (encoder, rate_override, denoiser, this_path, sr, g, p, q,))
                    jobs.append(job)
        
        for job in jobs:
            job.get()
        q.put(None)
        pool.close()
        pool.join()

    # TBD: Error handling could be improved.
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        print(err)
        exc_type, exc_obj, exc_tb = exc_info()
        fname = split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        raise
