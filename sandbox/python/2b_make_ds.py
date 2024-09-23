#! /usr/bin/env python
# Time-stamp: <2024-09-09 18:36:13 rytis>

# Create a credible event-gap dataset for individual rain records
# Hold out T = [10, 15) events and gaps as test, all the rest is train.
# This will generate datasets from the partitions

# The main resources used/generated
# 1. augment: intermediate TFRecords (required input)
# 2. ds: dataset TFRecords
# 3. config: dataset configuration (train/validation splitting)

import numpy as np
from os import mkdir
from os.path import isdir
from glob import glob

import rainml.ds
from rainml.ds import to_frames, BasicEventFrame, BasicEventFrameDatasetParser
from rainml.utils import check_bytes

# The features!
def logfreq(x): return np.reshape(-np.log10(x[:, 0]), (-1, 1))

# INPUTS
conf_basepath =  "./ds_conf"
aug_basepath = "./aug"

# OUTPUTS
ds_basepath =  "./ds"

target_freq = 44100
frame_lengths = [15]
dsids = [1, 3, 5, 4, 2]

tmp = f"{ds_basepath}/{target_freq}"
if not isdir(tmp):
    mkdir(tmp)
    print('creating:', tmp)

for fl in frame_lengths:
    tmp1 = f"{tmp}/{fl}"
    if not isdir(tmp1):
        mkdir(tmp1)
        print('creating:', tmp1)

for dsid in dsids:
    with open(conf_basepath + f"/{dsid}_impacts.npy", 'rb') as b:
        e_train = np.load(b)
        e_test = np.load(b)
        e = [e_train, e_test]
    with open(conf_basepath + f"/{dsid}_gaps.npy", 'rb') as b:
        g_train = np.load(b)
        g_test = np.load(b)
        g = [g_train, g_test]

    for fl in frame_lengths:
        print("frame length :", fl)
        aug = aug_basepath + f"/{dsid}_{target_freq}.tfrec"
        frame = BasicEventFrame(
            fl,
            event_hotspot=4, event_stride=1,  # oversampling events
            gap_stride=(fl//2),               # strided sampling of gaps
            transform=logfreq,                # the feature
            dtype='float16'                   # output in half-float
        )

        for _e, _g, name in zip(e, g, ['train', 'test']):
            cfg = [
                dict({'data': aug, 'events': _e, 'label': 1}),
                dict({'data': aug, 'events': _g, 'label': 0})
            ]
            ds_path = ds_basepath + f"/{target_freq}/{fl}/{dsid}_{name}.tfrec"
            to_frames(cfg, frame, ds_path, shuffle=1000, weights=[1.0/11.0, 10.0/11.0])
            # when we know that dataset is skewed we may 'help' to
            # improve the mixture by specifying weights.
        check_bytes(glob(ds_basepath + f"/{target_freq}/{fl}/{dsid}_*.tfrec"))
