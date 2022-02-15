import json, copy
from os import path
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from ray import tune
from functools import lru_cache
from tune_tf2.defaults import PBT_CSV, HPS_CSV, EXPLOIT_CSV

@lru_cache(maxsize=10)
def load_tune_df(experiment_dir):
    """Cache the creation of this dataframe. """
    return tune.Analysis(experiment_dir).dataframe()


def read_pbt_fitlog(pbt_dir, metrics_to_load=None, reconstruct=False):
    """Function for loading entire LF2 PBT fitlog.
    
    This function loads an entire PBT fitlog into a pandas DataFrame.
    Generations, trials, and epochs are labeled in their respective
    columns of the DataFrame. The DataFrame.pivot function is particularly
    useful for reshaping this data into whatever format is most useful.

    Parameters
    ----------
    pbt_dir : str
        The path to the PBT run.
    metrics_to_load : list of str, optional
        The metrics to load into the DataFrame, by default None loads 
        all of the available metrics.
    reconstruct : bool, optional
        Whether to reconstruct the actual paths of the individual models
        by duplicating paths of exploited models and removing the paths
        of killed models, by default False

    Returns
    pd.DataFrame
        A DataFrame containing the metrics of interest.

    Examples
    --------
    To load the PBT fitlog
    >>> fit_df = read_pbt_fitlog(pbt_dir)
    Pivot to show `val_nll_heldin` across epochs
    >>> plot_df = fit_df.pivot(
            index='epoch',
            columns='trial_id',
            values='val_nll_heldin')

    """

    if metrics_to_load is not None and 'epoch' not in metrics_to_load:
        metrics_to_load += ['epoch']
    tune_df = load_tune_df(pbt_dir)
    fit_dfs = []
    # Collect all of the LF2 data in `fit_dfs`
    for trial_id, logdir in zip(tune_df.trial_id, tune_df.logdir):
        if 'best_model' in logdir:
            continue
        train_data_path = path.join(
            logdir, 'model_dir', 'train_data.csv')
        fit_df = pd.read_csv(train_data_path, usecols=metrics_to_load)
        fit_df['trial_id'] = trial_id
        fit_dfs.append(fit_df)
    pbt_df = pd.concat(fit_dfs, ignore_index=True)
    # Get the generation info
    pbt_state_path = path.join(pbt_dir, PBT_CSV)
    pbt_state_df = pd.read_csv(
        pbt_state_path, usecols=['generation', 'epoch'])
    # add generation information to the dataframe
    prev_end = 0
    pbt_df['generation'] = -1
    for index, row in pbt_state_df.iterrows():
        gen, end_epoch = row.generation, row.epoch
        generation_rows = (pbt_df['epoch'] > prev_end) \
            & (pbt_df['epoch'] <= end_epoch)
        pbt_df.loc[generation_rows, 'generation'] = gen
        prev_end = end_epoch
    
    if reconstruct:
        return reconstruct_history(pbt_dir, pbt_df)
    else:
        return pbt_df


def plot_pbt_log(pbt_dir, 
                 plot_field, 
                 reconstruct=False, 
                 save_dir=None,
                 **kwargs):
    """Plots a specific field of the PBT fitlog.

    This function uses `load_pbt_fitlog` to load the training
    log and then overlays traces from all of the workers.
    It uses robust statistics to calculate the appropriate window
    for easy viewing.

    Parameters
    ----------
    pbt_dir : str
        The path to the PBT run.
    plot_field : str
        The metric to plot. See the fitlog headers or lfads_tf2
        source code for options.
    reconstruct : bool, optional
        Whether to reconstruct the actual paths of the individual models
        by duplicating paths of exploited models and removing the paths
        of killed models, by default False
    save_dir : str, optional
        The directory for saving the figure, by default None will
        show an interactive plot
    kwargs: optional
        Any keyword arguments to be passed to pandas.DataFrame.plot
    
    """

    # limit the metrics loaded from the CSV
    fit_df = read_pbt_fitlog(
        pbt_dir, [plot_field], reconstruct=reconstruct)
    plot_df = fit_df.pivot(
        index='epoch', columns='trial_id', values=plot_field)
    # compute the window
    epoch_range = plot_df.index.min(), plot_df.index.max()
    iqr = stats.iqr(plot_df, nan_policy='omit')
    median = np.nanmedian(plot_df)
    field_range = np.nanmin(plot_df), median + 5*iqr
    plot_kwargs = dict(
        xlim=epoch_range, 
        ylim=field_range,
        legend=False,
        alpha=0.3,
    )
    plot_kwargs.update(kwargs)
    # make the plot
    plot_df.plot(**plot_kwargs)
    plt.xlabel('epoch')
    plt.ylabel(plot_field)
    # save the plot if necessary
    if save_dir is not None:
        filename = plot_field.replace('.', '_').lower()
        fig_path = path.join(save_dir, f'{filename}.png')
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()


def read_pbt_hps(pbt_dir, reconstruct=False):
    """Function for loading the HPs used in a PBT run.

    This function loads the HPs used during a PBT run into a 
    pandas DataFrame. Generations and trials are labeled in 
    their respective columns of the DataFrame. The DataFrame.pivot 
    function is particularly useful for reshaping this data into 
    whatever format is most useful.

    Parameters
    ----------
    pbt_dir : str
        The path to the PBT run.
    reconstruct : bool, optional
        Whether to reconstruct the actual paths of the individual models
        by duplicating paths of exploited models and removing the paths
        of killed models, by default False

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the HPs of interest.

    Examples
    --------
    To load the PBT HPs
    >>> hps_df = read_pbt_hps(pbt_dir)
    Pivot to show `TRAIN.KL.CO_WEIGHT` across generations
    >>> plot_df = hps_df.pivot(
            index='generation',
            columns='trial_id',
            values='TRAIN.KL.CO_WEIGHT')
    """

    hps_path = path.join(pbt_dir, HPS_CSV)
    hps_df = pd.read_csv(hps_path)
    if reconstruct:
        return reconstruct_history(pbt_dir, hps_df)
    else:
        return hps_df


def plot_pbt_hps(pbt_dir, 
                 plot_field,
                 reconstruct=False,
                 save_dir=None,
                 **kwargs):
    """ Plots an HP for all models over the course of PBT.

    This function generates a plot to visualize how an HP
    changes over the course of PBT.

    Parameters
    ----------
    pbt_dir : str
        The path to the PBT run.
    plot_field : str
        The HP to plot. See the HP log headers or lfads_tf2
        source code for options.
    reconstruct : bool, optional
        Whether to reconstruct the actual paths of the individual 
        models by duplicating paths of exploited models and 
        removing the paths of killed models, by default False
    color : str, optional
        The color to use for the HP plot, passed to 
        matplotlib, by default 'b'
    alpha : float, optional
        The transparency for the HP plot traces, passed to 
        matplotlib, by default 0.2
    save_dir : str, optional
        The directory for saving the figure, by default None will
        show an interactive plot
    kwargs: optional
        Any keyword arguments to be passed to pandas.DataFrame.plot

    """

    hps_df = read_pbt_hps(pbt_dir, reconstruct=reconstruct)
    plot_df = hps_df.pivot(
        index='generation',
        columns='trial_id',
        values=plot_field)
    gen_range = plot_df.index.min(), plot_df.index.max()
    field_range = plot_df.min().min(), plot_df.max().max()
    plot_kwargs = dict(
        drawstyle='steps-post',
        legend=False,
        logy=True,
        c='b',
        alpha=0.2,
        title=f'{plot_field} for PBT run at {pbt_dir}',
        xlim=gen_range,
        ylim=field_range,
        figsize=(10,5)
    )
    plot_kwargs.update(kwargs)
    plot_df.plot(**plot_kwargs)
    if save_dir is None:
        plt.show()
    else:
        filename = plot_field.replace('.', '_').lower()
        fig_path = path.join(save_dir, f'{filename}.png')
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()


def reconstruct_history(pbt_dir, pbt_df):
    """Reconstructs actual model trajectories from fitlog.

    This function pieces together the successful models
    to determine what history looked like for the final 
    PBT models. Note that this function works on both 
    the PBT fitlog and HP DataFrames.

    Parameters
    ----------
    pbt_dir : str
        The path to the PBT run.
    pbt_df : pd.DataFrame
        A DataFrame containing the metrics or HPs of interest.

    Returns
    ------
    pd.DataFrame
        A DataFrame containing the reconstructed metrics
        or HPs of interest.
    """

    exploit_path = path.join(pbt_dir, EXPLOIT_CSV)
    exploit_df = pd.read_csv(exploit_path)
    exploits = exploit_df.pivot(
        index='generation',
        columns='old_trial',
        values='new_trial')
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
    int_cols = ['epoch', 'trial_id', 'generation']
    dtype_updates = {name: 'int64' for name in int_cols if name in recon_df}
    recon_df = recon_df.astype(dtype_updates)

    return recon_df
