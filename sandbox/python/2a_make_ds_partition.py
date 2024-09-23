#! /usr/bin/env python
# Time-stamp: <2024-09-09 17:25:27 rytis>

# Create a credible event-gap dataset for individual rain records
# Hold out T = [10, 15) events and gaps as test, all the rest is train.
# This will create the partition, but not the actual dataset.

# The main resources used/generated
# 1. txt (input): annotation files
# 2. conf (output): dataset configuration (train/validation splitting)

import numpy as np
from rainml.annotations import load_event_annotations

# INPUTS
inp_basepath = "./input"

# OUTPUTS
conf_basepath = "./ds_conf"

target_freq = 44100
frame_lengths = [15]
dsids = [1, 2, 3, 4, 5]

# Split and save events and gaps
for dsid in dsids:
    txt = inp_basepath + f"/{dsid}.txt"
    events = load_event_annotations(txt)
    # Test/Validation time interval: [10, 15)
    i1 = np.searchsorted(events, 10)
    i2 = np.searchsorted(events, 15)
    idx = np.arange(i1, i2)
    events_test = events[idx]
    events_train = np.delete(events, idx)

    # GAPS ====================================================================

    # Add 2ms margins to events
    evgaps = np.array([events[:-1]+0.002, events[1:]-0.002]).T
    # some intervals might be empty
    evgaps = evgaps[np.where(evgaps[:, 0] < evgaps[:, 1])[0]]

    # Test/Validation time interval: [10, 15)
    i1 = np.searchsorted(evgaps[:, 0], 10)
    i2 = np.searchsorted(evgaps[:, 1], 15)
    idx = np.arange(i1, i2)

    # Test:
    evgaps_test = evgaps[idx]
    # Train:
    evgaps_train = np.delete(evgaps, idx, axis=0)

    path = conf_basepath + f"/{dsid}_impacts.npy"
    with open(path, 'wb') as b:
        np.save(b, events_train, allow_pickle=True)
        np.save(b, events_test, allow_pickle=True)
        print(f"{path:12s} -> train:{events_train.shape[0]:3d} test:{events_test.shape[0]:3d}")

    path = conf_basepath + f"/{dsid}_gaps.npy"
    with open(path, 'wb') as b:
        np.save(b, evgaps_train, allow_pickle=True)
        np.save(b, evgaps_test, allow_pickle=True)
        print(f"{path:12s} -> train:{evgaps_train.shape[0]:3d} test:{evgaps_test.shape[0]:3d}")
