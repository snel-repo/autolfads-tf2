"""Functions for plotting comparisons of LFL and LF2 runs.

This module offers a few scripts for plotting the random
searches used to validate lfads_tf2. Running this module
as a script saves the most important plots to disk.
"""

import os
import logging
from os import path
from ray import tune
import pandas as pd
import numpy as np
from scipy import stats
from functools import lru_cache
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict
from tune_tf2.pbt.utils import load_tune_df
from lfads_tf2.compat import read_fitlog

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

SMTH_COLUMNS = [
    'smth_nll_heldin', 
    'smth_nll_heldout', 
    'smth_val_nll_heldin', 
    'smth_val_nll_heldout',
]

DEFAULT_SELECT_FN = lambda df: df['lfl_val_r2'] - 0.05 > df['lf2_val_r2']


def plot_compare_log(experiment_dir,
             plot_field, 
             select_fn=DEFAULT_SELECT_FN,
             logy=False,
             smooth=False,
             plot_all=True,
             save_dir=None):
    """Plots the training logs of paired LF2 and LFL runs.
    
    Parameters
    ----------
    experiment_dir : str
        The path to the validation run.
    plot_field : str
        The field to plot.
    select_fn : callable, optional
        A function that takes a pd.DataFrame and returns a boolean 
        series of the same length, by default DEFAULT_SELECT_FN
    logy: bool, optional
        Whether to plot a log transformation, by default False
    smooth: bool, optional
        Whether to plot smoothed data, by default False
    plot_all : bool, optional
        Whether to plot the unselected traces, by default True
    save_dir : str, optional
        Where to save the plot, by default None
    """
    # gather the data and select data to plot
    df = load_tune_df(experiment_dir)
    gp1_selection = select_fn(df)
    if plot_all:
        gp2_selection = ~gp1_selection 
    else:
        gp2_selection = ~df['config/TRAIN.TUNE_MODE']

    fig, ax = plt.subplots(figsize=(10,5))
    # keep track of the plotting boundaries
    ymin, ymax = np.inf, -np.inf
    xmin, xmax = np.inf, -np.inf
    datamax = -np.inf
    # plot each group separately with emphasis through transparency
    for selection, alpha in zip([gp1_selection, gp2_selection], [0.75, 0.1]):
        # select the desired runs
        sel_df = df[selection]
        lfl_dfs, lf2_dfs = [], []
        for logdir, trial_id in zip(sel_df.logdir, sel_df.trial_id):
            # limit the metrics loaded from the CSV
            metrics_to_load = [
                'epoch', # for x-values
                'smth_val_nll_heldin', # for finding endpoints
                plot_field, # for plotting
                'kl_wt', # for excluding ramping
                'l2_wt', # for excluding ramping
            ]
            # load LF2 training data
            lf2_csv_path = path.join(logdir, 'lf2/train_data.csv')
            lf2_df = pd.read_csv(lf2_csv_path, usecols=metrics_to_load)
            # load LFL training data
            lfl_model_dir = path.join(logdir, 'lfl')
            lfl_df = read_fitlog(lfl_model_dir, metrics_to_load)
            # add `trial_id` so we can concatenate the dataframes
            lf2_df['trial_id'] = trial_id
            lfl_df['trial_id'] = trial_id
            # accumulate these dataframes for the larger dataframe
            lfl_dfs.append(lfl_df)
            lf2_dfs.append(lf2_df)
        # compile all of the data into large, multi-run dataframes
        lfl_master_df = pd.concat(lfl_dfs)
        lf2_master_df = pd.concat(lf2_dfs)

        # ===== PLOT THE DATA =====
        # pivot dataframes to the plot field
        lfl_plot_df = lfl_master_df.pivot(
            index='epoch', columns='trial_id', values=plot_field)
        lf2_plot_df = lf2_master_df.pivot(
            index='epoch', columns='trial_id', values=plot_field)
        # perform calculations if needed
        if smooth:
            lfl_plot_df = lfl_plot_df.rolling(
                20, win_type='gaussian').mean(std=20)
            lf2_plot_df = lf2_plot_df.rolling(
                20, win_type='gaussian').mean(std=20)
        if logy:
            lfl_plot_df = np.log10(lfl_plot_df)
            lf2_plot_df = np.log10(lf2_plot_df)
        # plot the data
        lfl_plot_df.plot(label='LFL', color='b', alpha=alpha, ax=ax)
        lf2_plot_df.plot(label='LF2', color='r', alpha=alpha, ax=ax)
        
        # ===== PLOT THE ENDPOINTS =====
        # calculate the last ramping epochs
        kl_wt_df = lfl_master_df.pivot(
            index='epoch', columns='trial_id', values='kl_wt')
        l2_wt_df = lfl_master_df.pivot(
            index='epoch', columns='trial_id', values='l2_wt')
        last_ramp = pd.concat(
            [kl_wt_df.idxmax(), l2_wt_df.idxmax()], axis=1).max(axis=1)
        if not all(last_ramp == last_ramp.iloc[0]):
            logger.warning("Ramping is different across models. "
                "Model endpoints may be inaccurate.")
        last_ramp = last_ramp.iloc[0]
        # compute the indices with lowest smoothed validation NLL
        lfl_nll_df = lfl_master_df.pivot(
            index='epoch', columns='trial_id', values='smth_val_nll_heldin')
        lf2_nll_df = lf2_master_df.pivot(
            index='epoch', columns='trial_id', values='smth_val_nll_heldin')
        # find the endpoints
        lfl_end_epoch = lfl_nll_df[last_ramp:].idxmin()
        lf2_end_epoch = lf2_nll_df[last_ramp:].idxmin()
        lfl_end_value = lfl_plot_df.lookup(
            lfl_end_epoch.values, lfl_end_epoch.index)
        lf2_end_value = lf2_plot_df.lookup(
            lf2_end_epoch.values, lf2_end_epoch.index)
        # plot the endpoints
        plt.scatter(
            lfl_end_epoch, lfl_end_value, marker='x', color='b', alpha=alpha)
        plt.scatter(
            lf2_end_epoch, lf2_end_value, marker='x', color='r', alpha=alpha)

        # ===== CALCULATE PLOT WINDOW =====
        # Calculate the x-axis limits
        xmin = min([
            lfl_master_df.epoch.min(), 
            lf2_master_df.epoch.min(), 
            xmin])
        xmax = max([
            lfl_master_df.epoch.max(), 
            lf2_master_df.epoch.max(), 
            xmax])
        # Calculate the y-axis limits
        lfl_data = lfl_plot_df.to_numpy()
        lf2_data = lf2_plot_df.to_numpy()
        ymin = min([
            np.nanmin(lfl_data), 
            np.nanmin(lf2_data), 
            ymin])
        iqr = max([
            stats.iqr(lfl_data, nan_policy='omit'), 
            stats.iqr(lf2_data, nan_policy='omit')])
        median = np.nanmedian(np.concatenate([lfl_data, lf2_data]))
        # if the largest data point is smaller than the window, 
        # just use the largest data point
        datamax = max([
            np.nanmax(lfl_data),
            np.nanmax(lfl_data),
            datamax])
        ymax = min(max([median + 2 * iqr, ymax]), datamax)

    # add final touches to the plot
    ax.get_legend().remove()
    margin = 0.1 * iqr
    plt.axis([
        xmin, 
        xmax, 
        ymin - margin, 
        ymax + margin])
    plt.xlabel('epoch')
    field_label = plot_field
    if smooth:
        field_label = "smth_" + field_label
    if logy:
        field_label = "log_" + field_label
    plt.ylabel(field_label)
    plt.title("Train log for validation at: " + experiment_dir)
    # save or show the figure
    if save_dir == None:
        plt.show()
    else:
        filename = plot_field.replace('.', '_').lower()
        fig_path = path.join(save_dir, f'{filename}.png')
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()


def plot_compare_perf(experiment_dir, metric='val_r2', save_dir=None):
    """Plots a comparison of performance """
    lfl_label, lf2_label = 'lfl_'+metric, 'lf2_'+metric
    df = load_tune_df(experiment_dir)
    plt.scatter(df[lfl_label], df[lf2_label]); plt.plot([0,1], c='k')
    plt.xlabel(lfl_label); plt.ylabel(lf2_label)
    plt.title(f'validation run at {experiment_dir}')
    plt.xlim([0,1]); plt.ylim([0,1])
    plt.gca().set_aspect('equal', adjustable='box')
    if save_dir == None:
        plt.show()
    else:
        fig_path = path.join(save_dir, f'{metric}.png')
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()


def plot_timing(experiment_dir, save_dir=None):
    """Plots histograms of per-epoch times for LF2 """
    df = load_tune_df(experiment_dir)
    durations = {}
    hostnames = sorted(df.hostname.unique())
    for hostname in hostnames:
        durations[hostname] = []
        for logdir in df[df.hostname == hostname].logdir:
            lf2_log_path = path.join(logdir, 'lf2/train.log')
            with open(lf2_log_path, 'r') as lf2_log_file:
                end_next_line = False
                for line in lf2_log_file:
                    if 'Epoch: ' in line:
                        start_time = datetime.strptime(
                            line[:23], '%Y-%m-%d %H:%M:%S,%f')
                        end_next_line = True
                    elif end_next_line:
                        end_time = datetime.strptime(
                            line[:23], '%Y-%m-%d %H:%M:%S,%f')
                        duration = (end_time - start_time).total_seconds()
                        end_next_line = False
                        if duration < 10:
                            durations[hostname].append(duration)

    bins = np.histogram(
        np.hstack([data for data in durations.values()]), bins=100)[1]
    plt.figure(figsize=(10, 5))
    for hostname, deltas in durations.items():
        plt.hist(deltas, bins, alpha=0.5, label=hostname)
    plt.xlabel('LF2 epoch duration (s)')
    plt.title(f'compute times for validation run at {experiment_dir}')
    plt.legend()
    if save_dir == None:
        plt.show()
    else:
        fig_path = path.join(save_dir, 'timing.png')
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()


def plot_exp_compare(exp1_dir, 
                     exp2_dir, 
                     plot_field='lf2_val_r2', 
                     save_dir=None):
    """
    This function is intended to produce plots of a certain 
    field of `tune` data between two distinct experiments. 

    These include:
    'r2', 'val_r2', 'epoch', 'loss', 'step', 'val_loss', 
    'nll', 'val_nll', 'wt_kl', 'val_wt_kl', 'wt_l2', 
    'smth_nll', 'smth_val_nll', 'kl_wt', 'l2_wt', 'lr', 
    or 'lfl_gnorm'

    NOTE:
    - prepend 'lfl_' to the field for lfadslite or 'lf2_' 
        to the field for lfads_tf2
    - 'nll' and 'val_nll' are only available for LF2
    - 'step' is only available for LFL
    - 'wt_kl' and 'wt_l2' may be 'kl' and 'l2' respectively 
        for LFL (same for 'val_wt_kl')
    - 'kl_wt' and 'l2_wt' were introduced later for LF2 
        and so may not be available for older runs
    """
    # load dataframes from previous experiments
    df1 = load_tune_df(exp1_dir)
    df2 = load_tune_df(exp2_dir)

    # merge on configs to create a multi-experiment dataframe
    is_cfg_col = lambda name: 'config' in name and 'MODEL_DIR' not in name
    config_cols = [name for name in df1.columns if is_cfg_col(name) and name in df2]
    multiexp_df = df1.merge(df2, on=config_cols, suffixes=('_1', '_2'))

    # compare the field between experiments
    color = 'b' if 'lfl' in plot_field else 'r'
    plt.scatter(
        multiexp_df[plot_field + '_1'], 
        multiexp_df[plot_field + '_2'], 
        c=color)
    plt.plot([0,1], c='k')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel(path.basename(exp1_dir))
    plt.ylabel(path.basename(exp2_dir))
    plt.title(f'Comparison of {plot_field} from previous run')
    plt.gca().set_aspect('equal', adjustable='box')
    if save_dir == None:
        plt.show()
    else:
        fig_path = path.join(save_dir, 'inter_experiment.png')
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":

    # ---------- Select the experiment directory to plot ----------
    EXPERIMENT_DIR = path.join(
        '/snel/home/asedler/ray_results',
        'validation_withCon_fixedValidationLoss')
    
    # PREV_EXPERIMENT_DIR = ''
    PREV_EXPERIMENT_DIR = path.join(
        '/snel/home/asedler/ray_results',
        'validation_withCon_CD_refactorRerun')

    # ---------- Select which runs to plot ----------
    # select a group based on any criteria in the analysis dataframe, 
    # including configuration variables and final validation metrics. 
    # Selection variables must be boolean pandas series. 
    # Examine column names of 'df' for options.

    # selection of interest
    SELECT_FN = lambda df: df['lfl_val_r2'] - 0.05 > df['lf2_val_r2'] 
    PLOT_ALL = True

    # ---------- Select which field to plot ----------
    # can be any of 'loss', 'val_loss', 'nll_heldin', 'nll_heldout', 
    # 'val_nll_heldin', 'val_nll_heldout', 'wt_kl', 'val_wt_kl', 
    # 'wt_l2', 'lr', 'wt_ic_kl', 'val_wt_ic_kl', 'wt_co_kl', 
    # 'val_wt_co_kl', or 'gnorm'
    PLOT_FIELDS = [
        'loss',
        'val_loss',
        'nll_heldin',
        'val_nll_heldin',
        'val_wt_kl',
        'wt_l2',
        'lr',
        'gnorm',
    ]
    # choose fields to plot in log-space
    LOGY_FIELDS = ['lr', 'gnorm']
    # choose fields to smooth
    SMTH_FIELDS = ['gnorm']

    plot_save_dir = path.join(EXPERIMENT_DIR, 'analysis')
    os.makedirs(plot_save_dir, exist_ok=True)
    logger.info(f"Saving LFL validation plots in {plot_save_dir}")
    # Plot all of the logs
    for plot_field in PLOT_FIELDS:
        logger.info(f"Plotting `{plot_field}`")
        plot_compare_log(
            EXPERIMENT_DIR, 
            plot_field, 
            select_fn=SELECT_FN,
            logy=plot_field in LOGY_FIELDS,
            smooth=plot_field in SMTH_FIELDS,
            plot_all=PLOT_ALL,
            save_dir=plot_save_dir)
    # Plot decoding performance
    logger.info("Plotting performance comparison.")
    plot_compare_perf(EXPERIMENT_DIR, save_dir=plot_save_dir)
    # Plot per-epoch times
    logger.info("Plotting per-epoch times.")
    plot_timing(EXPERIMENT_DIR, save_dir=plot_save_dir)
    # Plot comparison to the previous experiment if it exists
    if PREV_EXPERIMENT_DIR:
        logger.info("Plotting comparison to previous run.")
        plot_exp_compare(
            PREV_EXPERIMENT_DIR,
            EXPERIMENT_DIR,
            save_dir=plot_save_dir)
    logger.info("LFL validation plotting complete.")
