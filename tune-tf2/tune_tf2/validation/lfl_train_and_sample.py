"""This script runs training and posterior sampling for an lfadslite model.

It runs the model based on HPs from a temporary JSON file. It is meant to 
be used in conjunction with `run_validation.py` to run LF2 and LFL models 
with identical HPs in parallel.
"""

import sys
import json
# make sure pbt_opt is on the path
sys.path.append('/snel/home/asedler/core/pbt-hp-opt')
from pbt_opt.lfads_wrapper.lfads_wrapper import lfadsWrapper

# load the JSON HPS fromt the configuration file
with open(sys.argv[1], 'r') as cfg_file:
    lfl_cfg = json.load(cfg_file)

# create the wrapper to manage an LFADS model
wrapper = lfadsWrapper()

# add the HPs for training that aren't found in LF2
lfl_cfg.update({
    'max_ckpt_to_keep': 5,
    'max_ckpt_to_keep_lve': 5,
    'ckpt_save_interval': 5,
    'csv_log': "fitlog",
    'output_filename_stem': "",
    'checkpoint_pb_load_name': "checkpoint",
    'checkpoint_name': "lfads_vae",
    'device': "gpu:0", # "cpu:0", or other gpus, e.g. "gpu:1"
    'ps_nexamples_to_process': int(1e8), # if larger than number of examples, process all
    'ic_enc_seg_len': 0,
    'do_reset_learning_rate': False,
    'do_train_encoder_only': False,
    'do_train_readin': True,
    'output_dist': 'poisson', # 'poisson' or 'gaussian'
    'do_causal_controller': False,
    'co_prior_var': 0.1, # NOT USED FOR AR PRIOR
    'do_train_prior_ar_atau': True,
    'do_train_prior_ar_nvar': True,
    'beta1': 0.9,
    'beta2': 0.999,
    'do_calc_r2': False,
    'valid_batch_size': lfl_cfg['batch_size'], # this is how it works in run_lfadslite if not specified
    'val_cost_for_pbt': 'heldout_trial',
})
# run the training
wrapper.train(
    hps_dict=lfl_cfg,
    lfads_save_path=lfl_cfg['lfads_save_dir'],
    ckpt_load_path='',
    epochs_per_generation=lfl_cfg['target_num_epochs'],
)
# update HPs for sampling
lfl_cfg.update({
    'kind': 'posterior_sample_and_average',
    'batch_size': 50, # number of samples to run for posterior sampling
    'checkpoint_pb_load_name': "checkpoint_lve",
})
# run posterior sampling
wrapper.posterior_mean_sample(
    hps_dict=lfl_cfg,
    ckpt_load_path=lfl_cfg['lfads_save_dir'],
)
