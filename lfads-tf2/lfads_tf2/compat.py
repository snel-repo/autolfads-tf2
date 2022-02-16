"""Utilities for interacting with older versions of LFADS.

This module contains functions that simplify the process
of getting fitlog and posterior sampling output from
individual lfadslite and original LFADS runs.
"""


import json
from collections import defaultdict
from glob import glob
from os import path

import h5py
import numpy as np
import pandas as pd
from lfads_tf2.tuples import SamplingOutput
from lfads_tf2.utils import load_data, merge_train_valid

DEFAULTS = {
    "adam_epsilon": 1e-08,
    "batch_size": 128,
    "beta1": 0.9,
    "beta2": 0.999,
    "cd_grad_passthru_prob": 0.0,
    "cell_clip_value": 5.0,
    "checkpoint_name": "lfads_vae",
    "checkpoint_pb_load_name": "checkpoint",
    "ci_enc_dim": 128,
    "ckpt_save_interval": 5,
    "co_dim": 1,
    "co_prior_var": 0.1,
    "con_dim": 128,
    "controller_input_lag": 1,
    "csv_log": "fitlog",
    "cv_keep_ratio": 1.0,
    "cv_rand_seed": 0.0,
    "data_dir": "/tmp/rnn_synth_data_v1.0/",
    "data_filename_stem": "chaotic_rnn_inputs_g1p5",
    "device": "gpu:0",
    "do_calc_r2": False,
    "do_causal_controller": False,
    "do_reset_learning_rate": False,
    "do_train_encoder_only": False,
    "do_train_prior_ar_atau": True,
    "do_train_prior_ar_nvar": True,
    "do_train_readin": True,
    "ext_input_dim": 0,
    "factors_dim": 50,
    "gen_dim": 200,
    "ic_dim": 64,
    "ic_enc_dim": 128,
    "ic_enc_seg_len": 0,
    "ic_post_var_min": 0.0001,
    "ic_prior_var": 0.1,
    "in_factors_dim": 0,
    "keep_prob": 0.95,
    "keep_ratio": 1.0,
    "kind": 1,
    "kl_co_weight": 1.0,
    "kl_ic_weight": 1.0,
    "kl_increase_epochs": 500,
    "kl_start_epoch": 0,
    "l2_ci_enc_scale": 0.0,
    "l2_con_scale": 0.0,
    "l2_gen_scale": 2000.0,
    "l2_ic_enc_scale": 0.0,
    "l2_increase_epochs": 500,
    "l2_start_epoch": 0,
    "learning_rate_decay_factor": 0.95,
    "learning_rate_init": 0.01,
    "learning_rate_n_to_compare": 6,
    "learning_rate_stop": 1e-05,
    "lfads_save_dir": "/tmp/lfads_chaotic_rnn_inputs_g1p5/lfadsOut/",
    "loss_scale": 10000.0,
    "max_ckpt_to_keep": 5,
    "max_ckpt_to_keep_lve": 5,
    "max_grad_norm": 200.0,
    "n_epochs_early_stop": 200,
    "output_dist": "poisson",
    "output_filename_stem": "",
    "prior_ar_atau": 10.0,
    "prior_ar_nvar": 0.1,
    "ps_nexamples_to_process": 100000000,
    "target_num_epochs": 0,
    "valid_batch_size": 128,
}

# A mapping from LF2 HPs to LFL HPs
HP_CONVERT = {
    # 'MODEL.ALIGN_MODE': False,
    "MODEL.CD_PASS_RATE": "cd_grad_passthru_prob",
    "MODEL.CD_RATE": "keep_ratio",  # cd rate is 1-keep_ratio
    "MODEL.CI_ENC_DIM": "ci_enc_dim",
    "MODEL.CON_DIM": "con_dim",
    "MODEL.CO_DIM": "co_dim",
    "MODEL.CO_PRIOR_NVAR": "prior_ar_nvar",
    "MODEL.CO_PRIOR_TAU": "prior_ar_atau",
    # 'MODEL.DATA_DIM': 100,
    "MODEL.DROPOUT_RATE": "keep_prob",  # dropout rate is 1-keep_prob
    "MODEL.EXT_INPUT_DIM": "ext_input_dim",
    "MODEL.FAC_DIM": "factors_dim",
    "MODEL.GEN_DIM": "gen_dim",
    "MODEL.IC_DIM": "ic_dim",
    "MODEL.IC_ENC_DIM": "ic_enc_dim",
    "MODEL.IC_POST_VAR_MIN": "ic_post_var_min",
    "MODEL.IC_PRIOR_VAR": "ic_prior_var",
    "MODEL.READIN_DIM": "in_factors_dim",
    # 'MODEL.SEQ_LEN': 50,
    "MODEL.SV_SEED": "cv_rand_seed",
    "MODEL.SV_RATE": "cv_keep_ratio",  # sv rate is 1-cv_keep_ratio
    "MODEL.CELL_CLIP": "cell_clip_value",
    "MODEL.CI_LAG": "controller_input_lag",
    "TRAIN.BATCH_SIZE": "batch_size",
    "TRAIN.DATA.DIR": "data_dir",
    "TRAIN.DATA.PREFIX": "data_filename_stem",
    "TRAIN.KL.CO_WEIGHT": "kl_co_weight",
    "TRAIN.KL.IC_WEIGHT": "kl_ic_weight",
    "TRAIN.KL.INCREASE_EPOCH": "kl_increase_epochs",
    "TRAIN.KL.START_EPOCH": "kl_start_epoch",
    "TRAIN.L2.CI_ENC_SCALE": "l2_ci_enc_scale",
    "TRAIN.L2.CON_SCALE": "l2_con_scale",
    "TRAIN.L2.GEN_SCALE": "l2_gen_scale",
    "TRAIN.L2.IC_ENC_SCALE": "l2_ic_enc_scale",
    "TRAIN.L2.INCREASE_EPOCH": "l2_increase_epochs",
    "TRAIN.L2.START_EPOCH": "l2_start_epoch",
    "TRAIN.LOSS_SCALE": "loss_scale",
    "TRAIN.LR.DECAY": "learning_rate_decay_factor",
    "TRAIN.LR.INIT": "learning_rate_init",
    "TRAIN.LR.PATIENCE": "learning_rate_n_to_compare",
    "TRAIN.LR.STOP": "learning_rate_stop",
    "TRAIN.ADAM_EPSILON": "adam_epsilon",
    "TRAIN.MAX_EPOCHS": "target_num_epochs",
    "TRAIN.MAX_GRAD_NORM": "max_grad_norm",
    "TRAIN.MODEL_DIR": "lfads_save_dir",
    # 'TRAIN.OVERWRITE': False,
    "TRAIN.PATIENCE": "n_epochs_early_stop",
    # 'TRAIN.TUNE_MODE': False,
    # 'TRAIN.USE_TB': False,
}

# the names of LFL logging files
FITLOG = "fitlog.csv"
SMTH_FITLOG = "fitlog_smoothed.csv"
GNORM = "gradnorms.csv"
ALIGN_FITLOG = "fitlog_genkl.csv"
ALIGN_GNORM = "gradnorms_genkl.csv"

# dictionary mapping to CSV file and column
LFL_METRIC_INFO = {
    "epoch": (FITLOG, 1),
    "step": (FITLOG, 3),
    "loss": (FITLOG, 5),
    "val_loss": (FITLOG, 6),
    "nll_heldin": (FITLOG, 8),
    "nll_heldout": (FITLOG, 9),
    "val_nll_heldin": (FITLOG, 10),
    "val_nll_heldout": (FITLOG, 11),
    "r2_heldin": (FITLOG, 14),
    "r2_heldout": (FITLOG, 15),
    "val_r2_heldin": (FITLOG, 16),
    "val_r2_heldout": (FITLOG, 17),
    "wt_kl": (FITLOG, 19),
    "val_wt_kl": (FITLOG, 20),
    "wt_l2": (FITLOG, 22),
    "kl_wt": (FITLOG, 24),
    "l2_wt": (FITLOG, 26),
    "lr": (FITLOG, 28),
    "wt_ic_kl": (FITLOG, 30),
    "val_wt_ic_kl": (FITLOG, 31),
    "wt_co_kl": (FITLOG, 33),
    "val_wt_co_kl": (FITLOG, 34),
    "smth_nll_heldin": (SMTH_FITLOG, 8),
    "smth_nll_heldout": (SMTH_FITLOG, 9),
    "smth_val_nll_heldin": (SMTH_FITLOG, 10),
    "smth_val_nll_heldout": (SMTH_FITLOG, 11),
    "gnorm": (GNORM, 0),
}
ORIG_LFADS_METRIC_INFO = {
    "epoch": (FITLOG, 1),
    "step": (FITLOG, 3),
    "loss": (FITLOG, 5),
    "val_loss": (FITLOG, 6),
    "nll_heldin": (FITLOG, 8),
    "val_nll_heldin": (FITLOG, 9),
    "wt_kl": (FITLOG, 11),
    "val_wt_kl": (FITLOG, 12),
    "wt_l2": (FITLOG, 14),
    "kl_wt": (FITLOG, 16),
    "l2_wt": (FITLOG, 18),
}
ALIGN_METRIC_INFO = {
    "epoch": (ALIGN_FITLOG, 1),
    "step": (ALIGN_FITLOG, 3),
    "kl": (ALIGN_FITLOG, 5),
    "val_kl": (ALIGN_FITLOG, 6),
    "gnorm": (ALIGN_GNORM, 0),
}


def read_hps(model_dir):
    """Read the hyperparameters from a given model directory.

    Parameters
    ----------
    model_dir : str
        The directory of the LFL model.

    Returns
    -------
    dict
        A dictionary of LFL hyperparameters
    """
    hp_pattern = path.join(model_dir, "hyperparameters-*.txt")
    hp_path = sorted(glob(hp_pattern))[0]
    with open(hp_path, "r") as hp_file:
        hps = json.load(hp_file)

    def convert_bools(val):
        if val == "true":
            return True
        elif val == "false":
            return False
        else:
            return val

    # convert json strings to booleans
    hps = {k: convert_bools(v) for k, v in hps.items()}

    return hps


def read_fitlog(model_dir, usecols=None, impl="lfadslite"):
    """Load a list of metrics from an older LFADS implementation.
    Compatible with lfadslite, the original implementation of
    LFADS from tensorflow_models, and lfadslite alignment.

    Parameters
    ----------
    model_dir : str
        The directory of the LFL model.
    usecols : list of str
        A list of desired metrics.
    impl : {'lfadslite', 'original', 'lfl-align'}
        The implementation that was used to run the model.

    Returns
    -------
    pd.DataFrame
        A dataframe of the loaded metrics.
    """

    if impl == "lfadslite":
        metric_info = LFL_METRIC_INFO
    elif impl == "original":
        metric_info = ORIG_LFADS_METRIC_INFO
    elif impl == "lfl-align":
        metric_info = ALIGN_METRIC_INFO
    else:
        raise AssertionError(f"Invalid implementation specified: {impl}")

    if usecols is None:
        usecols = metric_info.keys()
    # organize metrics and columns by file
    metrics = defaultdict(list)
    columns = defaultdict(list)
    for metric in usecols:
        file, column = metric_info[metric]
        metrics[file].append(metric)
        columns[file].append(column)
    dfs = []
    for file in metrics:
        # columns must be sorted for pandas to pair correctly
        sort_ixs = np.argsort(columns[file])
        sorted_metrics = np.array(metrics[file])[sort_ixs]
        sorted_columns = np.array(columns[file])[sort_ixs]
        # load dataframes from the individual files
        csv_path = path.join(model_dir, file)
        if path.isfile(csv_path):
            df = pd.read_csv(csv_path, names=sorted_metrics, usecols=sorted_columns)
            dfs.append(df)
    # merge the dataframes
    fit_df = pd.concat(dfs, axis=1)
    # convert all columns into numeric columns to get rid of 'NAN', etc.
    for col in fit_df:
        fit_df[col] = pd.to_numeric(fit_df[col], errors="coerce")
    return fit_df


def load_posterior_averages(model_dir, prefix="model_runs", merge_tv=False):
    """Loads the posterior sampling output from an older LFADS
    implementation. Compatible with both lfadslite and the original
    implementation of LFADS from tensorflow_models.

    Parameters
    ----------
    model_dir : str
        The directory of the LFL model.
    prefix : str, optional
        The prefix of the posterior sampling files to load, by
        default 'model_runs'.
    merge_tv : bool, optional
        Whether to merge training and validation data, by
        default False.

    Returns
    -------
    lfads_tf2.tuples.SamplingOutput or tuple
        A namedtuple containing the sampling output or a tuple
        of training and validation namedtuples. See SamplingOutput
        definition for more detail.
    """

    output_pattern = path.join(model_dir, prefix + "*")
    train_file, valid_file = sorted(glob(output_pattern))

    def get_h5_data(h5_path):
        with h5py.File(h5_path, "r") as h5file:
            # open the h5 file in a dictionary
            h5dict = {key: h5file[key][()] for key in h5file.keys()}
        return h5dict

    train_data = get_h5_data(train_file)
    valid_data = get_h5_data(valid_file)

    def make_sampling_output(h5dict):
        return SamplingOutput(
            rates=h5dict["output_dist_params"],
            factors=h5dict["factors"],
            gen_states=h5dict["gen_states"],
            gen_inputs=h5dict["controller_outputs"],
            gen_init=h5dict["gen_ics"],
            ic_post_mean=h5dict["post_g0_mean"],
            ic_post_logvar=h5dict["post_g0_logvar"],
            ic_prior_mean=h5dict["prior_g0_mean"],
            ic_prior_logvar=h5dict["prior_g0_logvar"],
        )

    if merge_tv:
        # Look for indices in the original data file
        hps_pattern = path.join(model_dir, "hyperparameters*.txt")
        hp_path = sorted(glob(hps_pattern))[-1]
        with open(hp_path, "r") as hp_file:
            hps = json.load(hp_file)
        data_dir = hps["data_dir"]
        prefix = hps["data_filename_stem"]
        try:
            # If there are index labels, use them to reassemble
            train_inds, valid_inds = load_data(data_dir, prefix=prefix, signal="inds")[
                0
            ]
            output = {}
            for key in train_data:
                output[key] = merge_train_valid(
                    train_data[key], valid_data[key], train_inds, valid_inds
                )
        except AssertionError:
            print(
                "No indices found for merge. "
                "Concatenating training and validation samples."
            )
            for key in train_data:
                output[key] = np.concatenate([train_data[key], valid_data[key]], axis=0)
        output = make_sampling_output(output)
    else:
        # keep the training and validation output separate
        train_output = make_sampling_output(train_data)
        valid_output = make_sampling_output(valid_data)
        output = train_output, valid_output

    return output
