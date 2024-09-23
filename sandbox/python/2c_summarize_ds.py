#! /usr/bin/env python
# Time-stamp: <2024-09-09 19:08:13 rytis>
# Create a credible event-gap dataset for individual rain records.
# This script summarizes dataset contents and provides some useful data metrics.
# Also, this illustrates the use of a parser, BasicEventFrameDatasetParser.
# 
# Input: dataset base folder containing datasets.
# Output: some printed text and a dsinfo.csv file.

import tensorflow as tf
import numpy as np
import pandas as pd
from os.path import basename
from rainml.ds import BasicEventFrameDatasetParser

# INPUTS
ds_basepath = "./ds"
target_freq = 44100
frame_lengths = [15]
dsids = [1, 3, 5, 4, 2]

# OUTPUTS:

def get_info(ds):
    num = np.zeros((2), dtype='int')
    for x, i in ds:
        assert not (np.any(np.isnan(x)) or np.any(np.isinf(np.abs(x)))), "invalid record"
        assert i in [0,1], "invalid label:" + str(i)
        num[i.numpy()] += 1
    tot = sum(num)
    prob_1 = num[1] / tot
    w = {0: 0.5*tot/num[0], 1: 0.5*tot/num[1]}
    print(f"impacts  : {num[1]:6d}")
    print(f"gaps     : {num[0]:6d}")
    print(f"sum total: {tot:6d}")
    print(f"P[impact]: {prob_1:.4f}")
    print(f"weights  : {w[0]:.3f} {w[1]:.3f}")
    print(f"imbalance ratio: {num[0]/num[1]:.2f}")
    return num[0], num[1], prob_1, w[0], w[1]

# Dataset parser
frame = BasicEventFrameDatasetParser()

for fl in frame_lengths:
    names = []
    ngaps = []
    wgaps = []
    nevts = []
    wevts = []
    probs = []
    for dsid in dsids:
        for ext in ['train', 'test']:
            name = f"/{target_freq}/{fl}/{dsid}_{ext}.tfrec"
            ds_path = ds_basepath + name
            print(f"-----------------------------------------------------------------------")
            print("integrity test for :", name, " (might take a long time)")
            ds = tf.data.TFRecordDataset([ds_path], num_parallel_reads=2).map(frame)
            ng, ne, p1, wg, we = get_info(ds)
            names.append(basename(ds_path))
            ngaps.append(ng)
            nevts.append(ne)
            probs.append(p1)
            wgaps.append(wg)
            wevts.append(we)
    
    df = pd.DataFrame(
        {
            "path": names,
            "total": np.array([a+b for a, b in zip(ngaps, nevts)], dtype=int),
            "n_gaps": np.array(ngaps, dtype=int),
            "n_events": np.array(nevts, dtype=int),
            "w_gaps": np.array(wgaps, dtype=float),
            "w_events": np.array(wevts, dtype=float),
            "prob": np.array(probs, dtype=float),
            "ratio": np.array([float(a/b) for a, b in zip(ngaps, nevts)], dtype=float)
        })
    # OUTPUT:
    info_path = ds_basepath + f"/{target_freq}/{fl}/dsinfo.csv"
    df.to_csv(info_path, float_format='%.3g')
    print("info-> ", info_path)
    print("done.")
