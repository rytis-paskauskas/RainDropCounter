#! /usr/bin/env python
# Time-stamp: <2024-02-02 09:11:21 rytis>
# -------------------------------------------------
# Extract events of raindrop hitting a surface, from audio

# import sys
# import stat
# import numpy as np
# import pandas as pd
# from functools import reduce
# from soundfile import read
# from keras.layers import AveragePooling1D, MaxPooling1D

# from datetime import datetime
# from glob import glob
# from os.path import basename, dirname
# from os import chmod

# sys.path.append("/home/rytis/PROJECTS/python_modules")
# import poincare as pc


import numpy as np
import csv

def load_event_annotations(fn):
    """Load a text file containing event listing. The input is expected as a 1-column csv file

    :param fn: path to a text file containing annotations
    :type fn: str
    """

    e = []
    with open(fn, 'r', encoding='utf-8') as f:
        rd = csv.reader(f, delimiter='\n', quoting=csv.QUOTE_NONE)
        # presumed header
        h = next(rd)[0]
        e = [float(t[0]) for t in rd]
        # In case thre was no header
        if h[0].isdigit():
            e.insert(0, float(h[0]))
    return np.array(e, dtype=float)


# # Not debugged from here on;
# def annotation_lock_event_annotations(fn):
#     """Make the file read-only.
#     To prevent accidental modifications. 
#     """
#     chmod(fn, stat.S_IREAD|stat.S_IRGRP)
#     return



# def annotation_identify_last_annotation_file(fid, basepath):
#     """Return the file name of the last date-time ordered file.
#     """
#     fn=None
#     fnl = glob(f"{basepath}/{fid}_*.txt")
#     if fnl:
#         fnl.sort()
#         fn=fnl[-1]
#     if not fn:
#         print('no matching files found')
#     return fn



# def annotation_clone_event_annotations(fn, fid):
#     """Load the last event file (indexed by datetime), save as a new
#     events file (identical to the previous) and locks the old file (no writing).

#     This is to be used in the event annotation refinement loop,
#     when an initial event file is already available:

#     BEGIN_LOOP
#     - clone the last annotated events file using clone_events
#     - plot the original data with events indicated (there is a routine
#       for that) for inspection of the events
    
#     - in case refinement is needed, open the last event file
#       (the one just cloned), edit the annotations, and save the file
#       with the same name.

#     END_LOOP

#     """
#     print('cloning from: ', basename(fn))
#     dn = dirname(fn)
#     e = annotation_load_event_annotations(fn)
#     annotation_lock_event_annotations(fn)
#     newid=datetime.now().isoformat()[:-7]
#     new_fn = f"{dn}/{fid}_{newid}.txt"
#     with open(new_fn, 'w') as f:
#         print('new ID: ', newid)
#         wr = csv.writer(f, delimiter='\n')
#         wr.writerow(['event list; cloned from ' + basename(fn) ])
#         wr.writerow(map( lambda t: format(t,'7.03f'), e))
#     return (e, new_fn)

# def annotation_save_annotations(annotated_t, filename, **kwargs):
#     """Save a list of events as annotation file
#     """
#     header = kwargs.pop('header', "Created on " + datetime.now().isoformat()[:-7] + " by annotation_save_annotations")
#     with open(filename, 'w', encoding='utf-8') as f:
#         f.write("# " + header + "\n")
#         for val in annotated_t:
#             f.write(f"{val:8.3f}\n")








