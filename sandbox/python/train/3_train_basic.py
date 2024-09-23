#! /usr/bin/env python
# Time-stamp: <2024-02-27 09:09:24 rytis>
#
import sys
import json
from configparser import ConfigParser
from os import makedirs, environ
from os.path import isfile, isdir

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import metrics, losses, optimizers, callbacks

from audio_encoder.frame_spec import FrameDatasetParser
from rainml.tfmodels import model_basic

environ['NCCL_DEBUG'] = 'INFO'

def compile_with_sgd(m: object, lrs: list, bndrs: list, metrics: list):
    lr_schedule = optimizers.schedules.PiecewiseConstantDecay(
        bndrs, lrs, name="StepWiseLearningRate"
    )
    m.compile(
        optimizer=optimizers.SGD(learning_rate=lr_schedule),
        loss=losses.BinaryCrossentropy(from_logits=True),
        metrics=metrics
    )
    return m

def basic_train(
        m: object,
        class_weight: dict,
        ds_train: tf.data.Dataset,
        ds_valid: tf.data.Dataset,
        batch_size: int,
        epochs: int):
    """Training protocol for V3 models.
    Model layers:
    0 : Input
    1 : Detector
    2 : DLogit
    """
    train_verbosity=2
    class Terminator(callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (logs['tp'] == 0) or (logs['tn'] == 0):
                self.model.stop_training = True

    info = m.fit(
        ds_train,
        # validation_data=ds_valid,
        batch_size=batch_size,
        epochs=epochs,
        class_weight=class_weight,
        # callbacks=[Terminator()],
        verbose=2
    )
    h = info.history
    tmp = m.evaluate(ds_valid, verbose=0)
    print(tmp)
    # F1 test
    if tmp[2] == 0:
        return None, h
    return m, h

def load_train_test_ds(basepath: str, dsids: list, desired_batch_size: int):
    """ basepath must contain a file dsinfo.csv for summary """
    assert isdir(basepath), basepath + "not a valid path "
    info_path = basepath + "/dsinfo.csv"
    assert isfile(info_path),  info_path + "not a file"

    info = pd.read_csv(info_path, index_col=0)
    cols = info.columns
    for t in ['path', 'n_gaps', 'n_events']:
        assert t in cols, t + " column not found in " + info_path
    frame = FrameDatasetParser()

    # Train dataset
    ds_paths = [f"{dsid}_train.tfrec" for dsid in dsids]
    info_train = info[info['path'].isin(np.array(ds_paths))]
    for t in info_train['path']:
        assert isfile(basepath + '/' + t), basepath + '/' + t + " not a valid file"
    # Loading the dataset
    ds_train = tf.data.TFRecordDataset(
        [basepath + '/' + t for t in info_train['path']], num_parallel_reads=4
    ).map(frame)
    # Computing the batche size and information
    n = np.array([sum(info_train['n_gaps']), sum(info_train['n_events'])], dtype='int')
    tot = np.sum(n)
    prob1 = n[1]/tot
    print(n)
    print(tot)
    print(prob1)
    min_batch_size = int(np.log(1.0-0.90)/np.log(1.0-prob1)) + 1
    weights = {0: 0.5*tot/n[0], 1: 0.5*tot/n[1]}

    batch_size = max(min_batch_size, desired_batch_size)

    print("Training features:")
    print(f"rain:  {n[1]:6d}")
    print(f"gaps:  {n[0]:6d}")
    print(f"total: {tot:6d}")
    print(f"prob1: {prob1:6.3f}")
    print(f"weights: {weights}")
    print(f"ratio: {n[0]/n[1]:.2f}")
    print(f"batch size={batch_size} number of batches: {tot//batch_size}")

    # Test dataset
    ds_paths = [f"{dsid}_test.tfrec" for dsid in dsids]
    info_test = info[info['path'].isin(np.array(ds_paths))]
    for t in info_test['path']:
        assert isfile(basepath + '/' + t), basepath + '/' + t + " not a valid file"
    # Load the dataset
    ds_test = tf.data.TFRecordDataset(
        [basepath + '/' + t for t in info_test['path']], num_parallel_reads=4
    ).map(frame)

    test_n = np.array([sum(info_test['n_gaps']), sum(info_test['n_events'])], dtype='int')
    test_tot = np.sum(test_n)

    return ds_train, ds_test, batch_size, weights

class Terminator(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (logs['tp'] == 0) or (logs['tn'] == 0):
            self.model.stop_training = True

def main():
    """ Read config and run training """

    # Do we have the config (ini) file?
    assert sys.argv and len(sys.argv) > 1, "configuration (ini) file not provided"
    cfg_path = sys.argv[1]
    assert isfile(cfg_path), f"not a file: {cfg_path}"
    config = ConfigParser()
    config.read(cfg_path)

    # Do we have a [naive_model] section?
    assert 'basic_model' in config.sections(), "[basic_model] section not found"
    cfg = config['basic_model']

    data_basepath = cfg['data_basepath']
    # proj_basepath = cfg['proj_basepath']
    out_basepath = cfg['out_basepath']
    # INPUTS

    frequency		= int(cfg['frequency'])
    frame_length 	= int(cfg['frame_length'])
    batch_size 		= int(cfg['batch_size'])
    epochs 		= int(cfg['epochs'])
    lr_thresholds 	= json.loads(cfg['lr_thresholds'])
    lr_boundaries 	= json.loads(cfg['lr_boundaries'])
    dsids 		= json.loads(cfg['ids'])
    model_N 		= int(cfg['model_N'])
    model_K 		= int(cfg['model_K'])
    model_FC            = json.loads(cfg['model_FC'])
    runs                = int(cfg['runs'])

    ds_basepath = data_basepath + f"/{frequency}/{frame_length}"
    assert isdir(ds_basepath), "invalid dataset base path (" + ds_basepath + " must be a valid path)"
    ds_train, ds_test, batch_size, w = load_train_test_ds(ds_basepath, dsids, batch_size)

    ds_train = ds_train.shuffle(10000, seed=42, reshuffle_each_iteration=True)\
    .batch(batch_size, drop_remainder=False, num_parallel_calls=8, deterministic=True)\
    .shuffle(100, seed=43, reshuffle_each_iteration=True).prefetch(1)

    ds_test = ds_test.shuffle(1000, seed=45, reshuffle_each_iteration=False)\
    .batch(batch_size, drop_remainder=False, num_parallel_calls=8, deterministic=True)\
    .shuffle(100, seed=46, reshuffle_each_iteration=False)

    # Create a dir for output
    makedirs(out_basepath, exist_ok=True)
    
    # Works with CPUs as well.
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("will attempt to use GPU: ")
        for t in gpus:
            print(t.name)

    strategy = tf.distribute.MirroredStrategy()

    # Retrain possibly multiple times, if the initial conditions are bad.
    for idx in range(runs):
        m = None
        print('run:', idx)
        with strategy.scope():
            METRICS = [
                metrics.TruePositives(name='tp'),
                metrics.TrueNegatives(name='tn'),
                metrics.FalsePositives(name='fp'),
                metrics.FalseNegatives(name='fn')
            ]
            m = model_basic(model_N, model_K, tuple(model_FC))
            m.compile(
                optimizer=optimizers.SGD(
                    learning_rate=optimizers.schedules.PiecewiseConstantDecay(lr_boundaries, lr_thresholds, name="StepWiseLearningRate")),
                metrics=METRICS,
                loss=losses.BinaryCrossentropy(from_logits=True)
            )
        print(m.summary())
        info = m.fit(
            ds_train,
            validation_data=ds_test,
            batch_size=batch_size,
            epochs=epochs,
            class_weight=w,
            callbacks=[Terminator()],
            verbose=2
        )

        m.save_weights(out_basepath + f'/weights_run{idx:02d}.h5')

        h = info.history
        df = pd.DataFrame.from_dict(h)
        df.to_csv(out_basepath + f'/history_run{idx:02d}.csv', sep=',', float_format='%.6g')


if __name__ == "__main__":
    sys.exit(main())
