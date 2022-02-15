"""Utilities for interacting with older versions of PBT.

This module contains functions that simplify the process 
of getting fitlog, hyperparameter, and posterior sampling 
output from lfadslite PBT runs.
"""

import json
import copy
import pandas as pd
from os import path
from glob import glob
from functools import partial
from multiprocessing import Pool
from collections import defaultdict
from lfads_tf2.compat import read_fitlog, load_posterior_averages


def _read_worker_fitlog(worker_dir, metrics=None):
    """Internal function for loading the fitlog of a single worker 
    during a single generation. Called in parallel by 
    `read_pbt_fitlog`.
    """

    gen, wid = path.basename(worker_dir).split('_')
    fitlog = read_fitlog(worker_dir, metrics)
    fitlog['trial_id'] = int(wid[1:])
    fitlog['generation'] = int(gen[1:])
    return fitlog


def _read_worker_hps(worker_dir):
    """ Internal function for loading the HPs of a single worker 
    during a single generation. Called in parallel by
    `read_pbt_hps`.
    """

    gen, wid = path.basename(worker_dir).split('_')
    hp_pattern = path.join(worker_dir, 'hyperparameters-*.txt')
    hp_path = sorted(glob(hp_pattern))[0]
    with open(hp_path, 'r') as hp_file:
        hps = json.load(hp_file)
    hps['trial_id'] = int(wid[1:])
    hps['generation'] = int(gen[1:])
    return hps


def read_pbt_fitlog(pbt_dir, metrics_to_load=None, reconstruct=False):
    """Function for loading entire LFL PBT fitlog. The `continuous` 
    option gets continuous logs for the final workers.
    """
    if metrics_to_load is not None and 'epoch' not in metrics_to_load:
        metrics_to_load += 'epoch'

    # Find all of the worker directories and infer generation numbers
    worker_dirs = sorted(glob(path.join(pbt_dir, 'g*w*')))
    # Set up the function and arguments to be executed in parallel
    func = partial(_read_worker_fitlog, metrics=metrics_to_load)
    args = worker_dirs
    # Load the fitlogs in parallel
    with Pool() as p:
        fitlogs = p.map(func, args)
    fit_df = pd.concat(fitlogs, ignore_index=True)
    if reconstruct:
        return reconstruct_history(pbt_dir, fit_df)
    else:
        return fit_df


def get_best_model_rates(pbt_dir, merge_tv=False):
    """Gets the rates for the best worker in an LFL run.
    """
    bw_path = path.join(pbt_dir, 'best_worker.done')
    with open(bw_path, 'r') as bw_file: 
        best_worker_path = bw_file.readline().strip()
    sampling_output = load_posterior_averages(
        best_worker_path, merge_tv=merge_tv)
    if merge_tv:
        return sampling_output.rates
    else:
        return sampling_output[0].rates, sampling_output[1].rates


def read_pbt_hps(pbt_dir, reconstruct=False):
    """Loads the dataframe of PBT HPs """
    # Load the LFL PBT HPs
    worker_dirs = sorted(glob(path.join(pbt_dir, 'g*w*')))
    # Set up the function and arguments to be executed in parallel
    func = _read_worker_hps
    args = worker_dirs
    # Load the HPs in parallel
    with Pool() as p:
        hps = p.map(func, args)
    hps_df = pd.DataFrame(hps)
    if reconstruct:
        return reconstruct_history(pbt_dir, hps_df)
    else:
        return hps_df


def reconstruct_history(pbt_dir, pbt_df):
    """Reconstructs actual history of the final PBT models for lfadslite.
    """
    # build the fitlog reconstruction dataframe
    decision_filepath = path.join(pbt_dir, 'decision_log.csv')
    decision_df = pd.read_csv(
        decision_filepath, 
        names=[
            'generation',
            'trial_id',
            'exploited',
            'competitor',
            'perf',
            'competitor_perf'], 
        usecols=[1, 3, 5, 7, 9, 11],
        dtype={'competitor': "Int64"})
    exploits = decision_df.pivot(
        index='generation', 
        columns='trial_id', 
        values='competitor')
    reconstruction = {i: [i] for i in range(len(exploits.columns))}
    for gen in range(1, len(exploits)):
        prev_reconst = copy.deepcopy(reconstruction)
        for exploited, exploiter in exploits.loc[gen].iteritems():
            if not pd.isna(exploiter):
                reconstruction[exploited] = copy.copy(prev_reconst[exploiter])
            reconstruction[exploited].append(exploited)
    recon_trials = pd.DataFrame(reconstruction).set_index(exploits.index)
    # reconstruct the fitlogs
    recon_df = pbt_df.copy()
    for gen in range(1, len(recon_trials+1)):
        for trial, worker in recon_trials.loc[gen].iteritems():
            ixs_to_overwrite = pbt_df[(pbt_df['generation'] == gen) \
                & (pbt_df['trial_id'] == trial)].index
            new_data = pbt_df[(pbt_df['generation'] == gen) \
                & (pbt_df['trial_id'] == worker)].copy()
            new_data['trial_id'] = trial
            recon_df.update(new_data.set_index(ixs_to_overwrite))
    # pd.DataFrame.update converted ints to floats, so bring them back
    recon_df = recon_df.astype({
        'epoch': 'int64', 
        'step': 'int64', 
        'trial_id': 'int64', 
        'generation': 'int64'})

    return recon_df
