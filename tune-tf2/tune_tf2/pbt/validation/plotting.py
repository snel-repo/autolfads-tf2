import os
import json
import logging
from ray import tune
from os import path
from glob import glob
import numpy as np
import pandas as pd
from scipy import stats
from functools import lru_cache

from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge

from lfads_tf2.utils import load_data, load_posterior_averages
from lfads_tf2.defaults import get_cfg_defaults
from lfads_tf2.compat import read_fitlog
from tune_tf2.pbt.validation.data import lfl_pbt_dirs, convert_hps
from tune_tf2.pbt.utils import load_tune_df, read_pbt_fitlog
from tune_tf2.pbt import compat
from tune_tf2.defaults import HPS_CSV


BIN_SIZE = 2 # bin size in ms (or downsampling factor)
DELAY = 90 # decoding delay in milliseconds

# set up the logger
logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

@lru_cache(maxsize=10)
def get_best_model_rates(model_dir, merge_tv=False):
    """Loads the rates for the best model in the PBT with caching."""
    try:
        # try to load from a posterior sampling file
        output = load_posterior_averages(model_dir, merge_tv=merge_tv)
    except FileNotFoundError:
        from lfads_tf2.utils import restrict_gpu_usage
        restrict_gpu_usage()
        from lfads_tf2.models import LFADS
        # create the LFADS model and do posterior sampling
        model = LFADS(model_dir=model_dir)
        model.sample_and_average(batch_size=10)
        output = load_posterior_averages(model_dir, merge_tv=merge_tv)
    if merge_tv:
        return output.rates
    else:
        return output[0].rates, output[1].rates

@lru_cache(maxsize=10)
def get_merged_data(data_dir, prefix, signal):
    """Fetches data from the h5 files with caching."""
    merged_data = load_data(
        data_dir, 
        prefix=prefix, 
        signal=signal, 
        merge_tv=True)[0]
    return merged_data


def plot_performance(pbt_group_dir, 
                     truth_available, 
                     alpha=1, 
                     merge_tv=False, 
                     save_dir=None):
    """Performs decoding on all of the best models and plots performance."""
    best_model_pattern = path.join(
        pbt_group_dir, 'sample_*', 'best_model', 'model_dir')
    model_dirs = sorted(glob(best_model_pattern))
    group_name = path.basename(pbt_group_dir)

    r2_data = defaultdict(list)
    for i, model_dir in enumerate(model_dirs):
        # get the path to the original data file
        cfg_path = path.join(model_dir, 'model_spec.yaml')
        config = get_cfg_defaults()
        config.merge_from_file(cfg_path)
        data_dir, prefix = config.TRAIN.DATA.DIR, config.TRAIN.DATA.PREFIX
        # find the corresponding LFL PBT run
        lfl_pbt_dir = lfl_pbt_dirs[group_name][i]
        # load the rates
        lfl_rates = compat.get_best_model_rates(lfl_pbt_dir, merge_tv=merge_tv)
        lf2_rates = get_best_model_rates(model_dir, merge_tv=merge_tv)

        if truth_available:
            # compare to the ground truth
            true_rates = load_data(
                data_dir, 
                prefix=prefix, 
                signal='truth', 
                merge_tv=merge_tv)[0]
            if merge_tv:
                # compute R2 over the entire dataset
                n_neurons = true_rates.shape[-1]
                r2_data['lfl_r2'].append(r2_score(
                    true_rates.reshape(-1, n_neurons), 
                    lfl_rates.reshape(-1, n_neurons)))
                r2_data['lf2_r2'].append(r2_score(
                    true_rates.reshape(-1, n_neurons), 
                    lf2_rates.reshape(-1, n_neurons)))
            else:
                # compute R2 over the train and valid sets separately
                true_train_rates, true_valid_rates = true_rates
                lfl_train_rates, lfl_valid_rates = lfl_rates
                lf2_train_rates, lf2_valid_rates = lf2_rates
                n_neurons = true_train_rates.shape[-1]
                r2_data['lfl_r2'].append(r2_score(
                    true_train_rates.reshape(-1, n_neurons), 
                    lfl_train_rates.reshape(-1, n_neurons)))
                r2_data['lf2_r2'].append(r2_score(
                    true_train_rates.reshape(-1, n_neurons), 
                    lf2_train_rates.reshape(-1, n_neurons)))
                r2_data['lfl_val_r2'].append(r2_score(
                    true_valid_rates.reshape(-1, n_neurons), 
                    lfl_valid_rates.reshape(-1, n_neurons)))
                r2_data['lf2_val_r2'].append(r2_score(
                    true_valid_rates.reshape(-1, n_neurons), 
                    lf2_valid_rates.reshape(-1, n_neurons)))
        else:
            # compare by decoding kinematics
            velocity = load_data(
                data_dir,
                prefix=prefix,
                signal='vel',
                merge_tv=merge_tv)[0]

            def evaluate_velocity_decoding(rates):
                n_bins = DELAY // BIN_SIZE
                if merge_tv:
                    # train and evaluate on all data
                    n_neurons = rates.shape[-1]
                    # get the data in the appropriate format
                    X = rates[:, :-n_bins, :].reshape(-1, n_neurons)
                    Y = velocity[:, n_bins:, :].reshape(-1, 2)
                    # train a classifier for the best regularized linear fit
                    regressor = Ridge(alpha=alpha, solver='lsqr')
                    regressor.fit(X, Y)
                    # evaluate the regression
                    Y_hat = regressor.predict(X)
                    r2 = r2_score(Y, Y_hat)
                    return r2
                else:
                    # separate the training and validation data
                    train_rates, valid_rates = rates
                    train_vel, valid_vel = velocity
                    n_neurons = train_rates.shape[-1]
                    # get the data in the appropriate format
                    X_train = train_rates[:, :-n_bins, :].reshape(-1, n_neurons)
                    Y_train = train_vel[:, n_bins:, :].reshape(-1, 2)
                    X_valid = valid_rates[:, :-n_bins, :].reshape(-1, n_neurons)
                    Y_valid = valid_vel[:, n_bins:, :].reshape(-1, 2)
                    # train a classifier for the best regularized linear fit
                    regressor = Ridge(alpha=alpha, solver='lsqr')
                    regressor.fit(X_train, Y_train)
                    # evaluate the regression
                    Y_hat_train = regressor.predict(X_train)
                    r2 = r2_score(Y_train, Y_hat_train)
                    Y_hat_valid = regressor.predict(X_valid)
                    val_r2 = r2_score(Y_valid, Y_hat_valid)

                    return r2, val_r2

            # find the best regularized linear fit
            lfl_r2 = evaluate_velocity_decoding(lfl_rates)
            lf2_r2 = evaluate_velocity_decoding(lf2_rates)

            if merge_tv:
                r2_data['lfl_r2'].append(lfl_r2)
                r2_data['lf2_r2'].append(lf2_r2)
            else:
                r2_data['lfl_r2'].append(lfl_r2[0])
                r2_data['lf2_r2'].append(lf2_r2[0])
                r2_data['lfl_val_r2'].append(lfl_r2[1])
                r2_data['lf2_val_r2'].append(lf2_r2[1])
    
    metrics = ['r2'] if merge_tv else ['r2', 'val_r2']
    df = pd.DataFrame.from_dict(r2_data)
    # calculate a good window
    points = df.stack()
    ctr = (points.max() + points.min()) / 2
    std = points.std()
    plot_range = np.clip([ctr - 4*std, ctr + 4*std], 0, 1)

    for metric in metrics:
        # plot the performance metrics
        plt.figure()
        lfl_label, lf2_label = 'lfl_'+metric, 'lf2_'+metric
        plt.scatter(df[lfl_label], df[lf2_label])
        plt.plot([0,1], c='k')
        plt.xlim(plot_range); plt.ylim(plot_range)
        if not truth_available:
            # clearly label "decoding" on the axes
            plt.xlabel('decoding_' + lfl_label)
            plt.ylabel('decoding_' + lf2_label)
        else:
            plt.xlabel(lfl_label)
            plt.ylabel(lf2_label)
        plt.title(f'PBT Validation Runs at {pbt_group_dir}')
        plt.grid()
        plt.gca().set_aspect('equal', adjustable='box')
        if save_dir is not None:
            # save the plots
            if not truth_available:
                filename = f'decoding_{metric}.png'
            else:
                filename = f'{metric}.png'
            fig_path = path.join(save_dir, filename)
            plt.savefig(fig_path, bbox_inches='tight')
            plt.close()

    if save_dir is None:
        plt.show()


def plot_best_nll(pbt_group_dir, 
                  nll_metric='val_nll_heldin', 
                  save_dir=None):
    """Plots the best smoothed validation NLL for 
    all runs in the group.
    """

    lf2_pbt_pattern = path.join(pbt_group_dir, 'sample_*')
    lf2_pbt_dirs = sorted(glob(lf2_pbt_pattern))
    group_name = path.basename(pbt_group_dir)

    nll_data = defaultdict(list)
    for i, lf2_pbt_dir in enumerate(lf2_pbt_dirs):
        metrics_to_load = ['epoch', nll_metric, 'kl_wt', 'l2_wt']
        lfl_pbt_dir = lfl_pbt_dirs[group_name][i]
        lfl_pbt_df = compat.read_pbt_fitlog(lfl_pbt_dir, metrics_to_load)
        lf2_pbt_df = read_pbt_fitlog(lf2_pbt_dir, metrics_to_load)
        # Pivot to NLL dataframes - NOTE: Generations are separated
        lfl_nll_df = pd.pivot_table(
            lfl_pbt_df, 
            index='epoch', 
            columns=['trial_id', 'generation'], 
            values=nll_metric)
        lf2_nll_df = pd.pivot_table(
            lf2_pbt_df, 
            index='epoch', 
            columns=['trial_id', 'generation'], 
            values=nll_metric)
        # Use exponential smoothing on the data (wasn't logged for LFL runs)
        lfl_smth_nll_df = lfl_nll_df.ewm(alpha=0.7).mean()
        lf2_smth_nll_df = lf2_nll_df.ewm(alpha=0.7).mean()
        # Exclude ramping steps
        df = lfl_pbt_df
        first_postramp_epoch = df[(df.kl_wt == 1) & (df.l2_wt == 1)].epoch.min()
        # Find the minimum NLL values
        nll_data[f'lfl_smth_{nll_metric}'].append(
            lfl_smth_nll_df[first_postramp_epoch:].min().min())
        nll_data[f'lf2_smth_{nll_metric}'].append(
            lf2_smth_nll_df[first_postramp_epoch+1:].min().min())

    df = pd.DataFrame.from_dict(nll_data)
    # plot the performance metric
    plt.figure()
    lfl_label = 'lfl_smth_' + nll_metric
    lf2_label = 'lf2_smth_' + nll_metric
    plt.scatter(df[lfl_label], df[lf2_label])
    plt.plot([0,1], c='k')
    points = pd.concat([df[lfl_label], df[lf2_label]], ignore_index=True)
    ctr = (points.max() + points.min()) / 2
    std = points.std()
    plt.xlim([ctr - 2*std, ctr + 2*std])
    plt.ylim([ctr - 2*std, ctr + 2*std])
    plt.xlabel(lfl_label); plt.ylabel(lf2_label)
    plt.title(f'PBT Validation Runs at {pbt_group_dir}')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    plt.xticks(rotation=45, ha='right')

    if save_dir is None:
        plt.show()
    else:
        fig_path = path.join(save_dir, f'{nll_metric}.png')
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()


def plot_metric(pbt_run_dir, plot_field, save_dir=None):
    """ Plots the training logs for LFL and LF2 """
    lf2_pbt_dir = pbt_run_dir
    data_group = path.basename(path.dirname(pbt_run_dir))
    sample_num = int(path.basename(pbt_run_dir).split('_')[-1])
    lfl_pbt_dir = lfl_pbt_dirs[data_group][sample_num]

    # limit the metrics loaded from the CSV
    metrics_to_load = ['epoch', plot_field]

    # Collect all of the LF2 data in `lf2_dfs`
    tune_df = load_tune_df(lf2_pbt_dir)
    lf2_dfs = {}
    for wid, logdir in zip(tune_df.trial_id, tune_df.logdir):
        train_data_path = path.join(logdir, 'model_dir', 'train_data.csv')
        lf2_df = pd.read_csv(train_data_path, usecols=metrics_to_load)
        lf2_dfs[wid] = lf2_df

    # Collect all of the LFL data in `lfl_dfs`
    worker_dirs = sorted(glob(path.join(lfl_pbt_dir, 'g*w*')))
    gen_and_worker_ids = [path.basename(d).split('_') for d in worker_dirs]
    lfl_dfs = defaultdict(list)
    for worker_dir, (gen, wid) in zip(worker_dirs, gen_and_worker_ids):
        # load the fitlog
        lfl_df = read_fitlog(worker_dir, metrics_to_load)
        lfl_dfs[wid].append(lfl_df)
    lfl_dfs = {wid: pd.concat(df_list, ignore_index=True) for wid, df_list in lfl_dfs.items()}
    
    # Calculate the x-axis limits
    lfl_max_epoch = max([df.epoch.max() for df in lfl_dfs.values()])
    lf2_max_epoch = max([df.epoch.max() for df in lf2_dfs.values()])
    x_max = max([lfl_max_epoch, lf2_max_epoch])
    lfl_min_epoch = min([df.epoch.min() for df in lfl_dfs.values()])
    lf2_min_epoch = min([df.epoch.min() for df in lf2_dfs.values()])
    x_min = min([lfl_min_epoch, lf2_min_epoch])
    # Calculate the y-axis limits
    lfl_min_plot_field = min([df[plot_field].min() for df in lfl_dfs.values()])
    lf2_min_plot_field = min([df[plot_field].min() for df in lf2_dfs.values()])
    y_min = min([lfl_min_plot_field, lf2_min_plot_field])
    lfl_field_data = np.concatenate([df[plot_field] for df in lfl_dfs.values()])
    lf2_field_data = np.concatenate([df[plot_field] for df in lf2_dfs.values()])
    lfl_iqr = stats.iqr(lfl_field_data)
    lf2_iqr = stats.iqr(lf2_field_data)
    iqr = max([lfl_iqr, lf2_iqr])
    median = np.nanmedian(np.concatenate([lfl_field_data, lf2_field_data]))
    y_max = median + 5 * iqr
    # Plot the data
    plt.figure(figsize=(17, 6))
    plt.suptitle(f'PBT Validation at {lf2_pbt_dir}')
    plt.subplot(1, 2, 1)
    for df in lfl_dfs.values(): 
        df[plot_field].plot(xlim=(x_min, x_max), ylim=(y_min, y_max))
    plt.xlabel('epoch')
    plt.ylabel(plot_field)
    plt.title('LFL PBT')
    plt.subplot(1, 2, 2)
    for df in lf2_dfs.values():
        df[plot_field].plot(xlim=(x_min, x_max), ylim=(y_min, y_max))
    plt.xlabel('epoch')
    plt.ylabel(plot_field)
    plt.title('LF2 PBT')

    if save_dir is None:
        plt.show()
    else:
        filename = plot_field.replace('.', '_').lower()
        fig_path = path.join(save_dir, f'{filename}.png')
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()


def plot_hps(pbt_run_dir, plot_field, color='b', alpha=0.2, save_dir=None):
    """Plots the hyperparameter trajectories for LFL and LF2."""
    lf2_pbt_dir = pbt_run_dir
    data_group = path.basename(path.dirname(pbt_run_dir))
    sample_num = int(path.basename(pbt_run_dir).split('_')[-1])
    lfl_pbt_dir = lfl_pbt_dirs[data_group][sample_num]

    # Load the LF2 PBT HPs
    lf2_df_path = path.join(lf2_pbt_dir, HPS_CSV)
    lf2_df = pd.read_csv(lf2_df_path)
    # Load the LFL PBT HPs
    worker_dirs = sorted(glob(path.join(lfl_pbt_dir, 'g*w*')))
    gen_and_worker_ids = [path.basename(d).split('_') for d in worker_dirs]
    lfl_hps = defaultdict(list)
    for worker_dir, (gen, wid) in zip(worker_dirs, gen_and_worker_ids):
        hp_path = sorted(glob(path.join(worker_dir, 'hyperparameters-*.txt')))[0]
        with open(hp_path, 'r') as hp_file:
            hps = json.load(hp_file)
        lfl_hps['generation'].append(int(gen[1:]))
        lfl_hps['trial'].append(int(wid[1:]))
        for name, value in hps.items():
            lfl_hps[name].append(value)
    lfl_df = pd.DataFrame.from_dict(lfl_hps)
    lfl_plot_field = convert_hps[plot_field]
    # calculate the appropriate axis limits
    x_max = max([lfl_df['generation'].max(), lf2_df['generation'].max()])
    x_min = min([lfl_df['generation'].min(), lf2_df['generation'].min()])
    y_max = max([lfl_df[lfl_plot_field].max(), lf2_df[plot_field].max()])
    y_min = min([lfl_df[lfl_plot_field].min(), lf2_df[plot_field].min()])
    # Plot the data
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(17, 6))
    plt.suptitle(f'PBT Validation at {lf2_pbt_dir}')
    # use pivot tables to aggregate the data appropriately
    lfl_plot_df = lfl_df.pivot(index='generation', columns='trial', values=lfl_plot_field)
    lf2_plot_df = lf2_df.pivot(index='generation', columns='trial', values=plot_field)
    lfl_plot_df.plot(
        drawstyle="steps-post", 
        legend=False, 
        logy=True, 
        c=color, 
        alpha=alpha,
        title='LFL PBT',
        ylim=(y_min, y_max),
        xlim=(x_min, x_max),
        ax=ax1)
    ax1.set_ylabel(lfl_plot_field)
    lf2_plot_df.plot(
        drawstyle="steps-post", 
        legend=False, 
        logy=True, 
        c=color, 
        alpha=alpha,
        title='LF2 PBT',
        ylim=(y_min, y_max),
        xlim=(x_min, x_max),
        ax=ax2)
    ax2.set_ylabel(plot_field)
    if save_dir is None:
        plt.show()
    else:
        filename = plot_field.replace('.', '_').lower()
        fig_path = path.join(save_dir, f'{filename}.png')
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()


def plot_rates(pbt_dir, train_rates=False, save_dir=None):
    """Plots samples of rates of the best models."""
    model_dir = path.join(
        pbt_dir,
        'best_model',
        'model_dir')
    # get the path to the original data file
    cfg_path = path.join(model_dir, 'model_spec.yaml')
    config = get_cfg_defaults()
    config.merge_from_file(cfg_path)
    data_dir, prefix = config.TRAIN.DATA.DIR, config.TRAIN.DATA.PREFIX
    # find the corresponding LFL PBT run
    group_name = path.basename(path.dirname(pbt_dir))
    sample_num = int(path.basename(pbt_dir).split('_')[-1])
    lfl_pbt_dir = lfl_pbt_dirs[group_name][sample_num]
    lfl_rates = compat.get_best_model_rates(lfl_pbt_dir, merge_tv=False)
    lf2_rates = get_best_model_rates(model_dir, merge_tv=False)
    spikes = load_data(data_dir, prefix=prefix, signal='data')[0]
    if train_rates:
        # select the training data
        lf2_rates = lf2_rates[0]
        lfl_rates = lfl_rates[0]
        spikes = spikes[0]
    else:
        # select the validation data
        lf2_rates = lf2_rates[1]
        lfl_rates = lfl_rates[1]
        spikes = spikes[1]
    # normalize the spikes
    normed_spikes = np.clip(spikes, 0, 1)
    # use a function to normalize the rates
    def norm_and_clip(rates):
        # compute mean and standard deviations across time dimension
        std = np.std(rates, axis=1, keepdims=True)
        mean = np.mean(rates, axis=1, keepdims=True)
        return np.clip((rates - mean) / std, -4, 4)
    # normalize the rates
    normed_lfl_rates = norm_and_clip(lfl_rates)
    normed_lf2_rates = norm_and_clip(lf2_rates)
    # compute difference
    diff_rates = np.clip(
        normed_lf2_rates - normed_lfl_rates, -4, 4)
    # get indices that will sort the neurons by rate peaks
    lfl_peaks = np.argmax(normed_lfl_rates, axis=1)
    lfl_ixs = np.expand_dims(np.argsort(lfl_peaks), 1)
    lf2_peaks = np.argmax(normed_lf2_rates, axis=1)
    lf2_ixs = np.expand_dims(np.argsort(lf2_peaks), 1)
    # use a function to sort and plot rates
    def sort_and_plot(data, title, plt_ix, sort_ixs, spikes=False, cmap='viridis'):
        # sort neurons by peaks
        sorted_data = np.take_along_axis(data, sort_ixs, axis=-1)
        plt.subplot(2, 2, plt_ix, title=title)
        if spikes:
            plt.imshow(sorted_data[0].T, cmap=cmap)
        else:
            plt.imshow(sorted_data[0].T, vmin=-4, vmax=4, cmap=cmap)
        plt.colorbar()
    # add data info to plot title
    plt.figure(figsize=(10, 6))
    data_tag = path.join(*data_dir.split('/')[-2:])
    n_neurons = lfl_rates.shape[-1]
    rate_r2 = r2_score(
        lfl_rates.reshape(-1, n_neurons),
        lf2_rates.reshape(-1, n_neurons))
    plt.suptitle(f'PBT Validation for Data at {data_tag} - '
        f'Rate R2: {rate_r2:.2f}')
    # sort and plot rates and spikes
    sort_and_plot(
        normed_lfl_rates, 'LFL Normalized Rates', 1, lfl_ixs)
    sort_and_plot(
        normed_lf2_rates, 'LF2 Normalized Rates', 2, lf2_ixs)
    sort_and_plot(
        normed_spikes, 'Binary Spikes', 3, lfl_ixs, spikes=True)
    sort_and_plot(
        diff_rates, 'Normalized Difference', 4, lfl_ixs, cmap='coolwarm')
    if save_dir is None:
        plt.show()
    else:
        if train_rates:
            filename = 'train_rates.png'
        else:
            filename = 'valid_rates.png'
        rates_fig_path = path.join(save_dir, filename)
        plt.savefig(rates_fig_path, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":

    # ========== PLOTTING CONFIGURATION ==========
    PBT_HOME = path.join('/snel/home/asedler/ray_results',
        'pbt_validation_allImprovements')
    DATA_GROUP = 'chaotic/chaotic_05_replications'
    TRUTH_AVAILABLE = True
    MERGE_TV = False
    ANALYSIS_NAME = 'analysis'
    HPS = [
        'TRAIN.LR.INIT',
        'MODEL.DROPOUT_RATE',
        'MODEL.CD_RATE',
        'TRAIN.KL.IC_WEIGHT',
        'TRAIN.KL.CO_WEIGHT',
        'TRAIN.L2.CON_SCALE',
        'TRAIN.L2.GEN_SCALE',
    ]
    METRICS = [
        'nll_heldin',
        'val_nll_heldin',
    ]
    # ============================================

    pbt_group_dir = path.join(PBT_HOME, DATA_GROUP)
    # create a directory to save the plots
    plot_save_dir = path.join(pbt_group_dir, ANALYSIS_NAME)
    os.makedirs(plot_save_dir, exist_ok=True)
    logger.info(f'Saving plots for PBT validation to {plot_save_dir}.')
    # find all of the PBT runs in this group
    pbt_pattern = path.join(pbt_group_dir, 'sample_*')
    pbt_run_dirs = sorted(glob(pbt_pattern))
    logger.info(f'Found {len(pbt_run_dirs)} PBT run(s).')
    # generate all of the plots
    for pbt_run_dir in pbt_run_dirs:
        logger.info(f'Plotting run at {pbt_run_dir}.')
        # make the plotting directory
        run_name = path.basename(pbt_run_dir)
        run_plot_dir = path.join(plot_save_dir, run_name)
        os.makedirs(run_plot_dir, exist_ok=True)
        # plot rates for training and validation data
        plot_rates(pbt_run_dir, train_rates=True, save_dir=run_plot_dir)
        plot_rates(pbt_run_dir, train_rates=False, save_dir=run_plot_dir)
        for hp in HPS:
            # plot HPs
            plot_hps(pbt_run_dir, hp, save_dir=run_plot_dir)
        for metric in METRICS:
            # plot metrics
            plot_metric(pbt_run_dir, metric, save_dir=run_plot_dir)
    # compare with ground truth or decode for R2
    plot_performance(
        pbt_group_dir, 
        TRUTH_AVAILABLE, 
        merge_tv=MERGE_TV,
        save_dir=plot_save_dir)
    # plot best losses
    for nll_metric in ['nll_heldin', 'val_nll_heldin']:
        plot_best_nll(
            pbt_group_dir, 
            nll_metric=nll_metric, 
            save_dir=plot_save_dir)

    logger.info(f'PBT validation plotting complete.')
