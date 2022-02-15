import yaml, os, h5py, sys, shutil, subprocess, tempfile, json, ray
from yacs.config import CfgNode as CN
from sklearn.metrics import r2_score
from glob import glob
from ray import tune
from os import path
import pandas as pd
import numpy as np

from lfads_tf2.defaults import get_cfg_defaults
from lfads_tf2.utils import load_data, flatten, unflatten, load_posterior_averages

EXPERIMENT_NAME = 'validation_withCon_fixedValidationLoss'
NUM_SAMPLES = 70
SHARE_INIT = False
RELATIVE_CFG_PATH = 'config/validation.yaml'
# Specify the location of a previous validation run with identical 
# HPs, if applicable. The `tuneValidation` class will pull LFL metrics 
# from here instead of training a completely new model. Set this to 
# an empty string if no previous run is available.
PREV_EXPERIMENT = '/snel/home/asedler/ray_results/validation_withCon_CD_refactorRerun'

# start the cluster
ray.init(redis_address="localhost:6379")

# specify HPs whose initial values will be overwritten by samples
np.random.seed(seed=0) # seed the randomly sampled configs
cfg_samples = {
    'MODEL.DROPOUT_RATE': tune.uniform(0.0, 0.7),
    'MODEL.CD_RATE': tune.uniform(0.3, 0.7),
    'MODEL.CD_PASS_RATE': 0.0, # tune.uniform(0, 1),
    'MODEL.SV_RATE': 0.0, # 0.1,
    'TRAIN.LR.INIT': 0.01,
    'TRAIN.KL.IC_WEIGHT': tune.loguniform(2e-5, 1e-3),
    'TRAIN.KL.CO_WEIGHT': tune.loguniform(2e-5, 2e-3),
    'TRAIN.L2.IC_ENC_SCALE': 0.0, # tune.loguniform(1e-5, 1e-3),
    'TRAIN.L2.CI_ENC_SCALE': 0.0, # tune.loguniform(1e-5, 1e-3),
    'TRAIN.L2.GEN_SCALE': tune.loguniform(2e-4, 6e0),
    'TRAIN.L2.CON_SCALE': tune.loguniform(2e-4, 6e0),
    'TRAIN.BATCH_SIZE': 200,
    'TRAIN.TUNE_MODE': True,
    'TRAIN.PBT_MODE': False,
}

# specify the default HPs and architecture
validation_dir = path.dirname(path.abspath(__file__))
cfg_path = path.join(validation_dir, RELATIVE_CFG_PATH)

# store the analysis dataframe so it is accessible to all of the
if PREV_EXPERIMENT:
    print(f"Searching for matching LFL models at {PREV_EXPERIMENT}.")
    prev_exp_df = tune.Analysis(PREV_EXPERIMENT).dataframe()
    resources = {"cpu": 2, "gpu": 0.5} # only need 1/2 GPU per model
else:
    print("No `PREV_EXPERIMENT` specified. Training new LFL models.")
    resources = {"cpu": 5, "gpu": 1.0} # set aside more GPU space for LFL

# converts HP names from lf2 to lfl - commented HPs are not used
convert = {
    # 'MODEL.ALIGN_MODE': False,
    'MODEL.CD_PASS_RATE': 'cd_grad_passthru_prob',
    'MODEL.CD_RATE': 'keep_ratio',
    'MODEL.CI_ENC_DIM': 'ci_enc_dim',
    'MODEL.CON_DIM': 'con_dim',
    'MODEL.CO_DIM': 'co_dim',
    'MODEL.CO_PRIOR_NVAR': 'prior_ar_nvar',
    'MODEL.CO_PRIOR_TAU': 'prior_ar_atau',
    # 'MODEL.DATA_DIM': 100,
    'MODEL.DROPOUT_RATE': 'keep_prob',
    'MODEL.EXT_INPUT_DIM': 'ext_input_dim',
    'MODEL.FAC_DIM': 'factors_dim',
    'MODEL.GEN_DIM': 'gen_dim',
    'MODEL.IC_DIM': 'ic_dim',
    'MODEL.IC_ENC_DIM': 'ic_enc_dim',
    'MODEL.IC_POST_VAR_MIN': 'ic_post_var_min',
    'MODEL.IC_PRIOR_VAR': 'ic_prior_var',
    'MODEL.READIN_DIM': 'in_factors_dim',
    # 'MODEL.SEQ_LEN': 50,
    'MODEL.SV_SEED': 'cv_rand_seed',
    'MODEL.SV_RATE': 'cv_keep_ratio',
    'MODEL.CELL_CLIP': 'cell_clip_value',
    'MODEL.CI_LAG': 'controller_input_lag',
    'TRAIN.BATCH_SIZE': 'batch_size',
    'TRAIN.DATA.DIR': 'data_dir',
    'TRAIN.DATA.PREFIX': 'data_filename_stem',
    'TRAIN.KL.CO_WEIGHT': 'kl_co_weight',
    'TRAIN.KL.IC_WEIGHT': 'kl_ic_weight',
    'TRAIN.KL.INCREASE_EPOCH': 'kl_increase_epochs',
    'TRAIN.KL.START_EPOCH': 'kl_start_epoch',
    'TRAIN.L2.CI_ENC_SCALE': 'l2_ci_enc_scale',
    'TRAIN.L2.CON_SCALE': 'l2_con_scale',
    'TRAIN.L2.GEN_SCALE': 'l2_gen_scale',
    'TRAIN.L2.IC_ENC_SCALE': 'l2_ic_enc_scale',
    'TRAIN.L2.INCREASE_EPOCH': 'l2_increase_epochs',
    'TRAIN.L2.START_EPOCH': 'l2_start_epoch',
    'TRAIN.LOSS_SCALE': 'loss_scale',
    'TRAIN.LR.DECAY': 'learning_rate_decay_factor',
    'TRAIN.LR.INIT': 'learning_rate_init',
    'TRAIN.LR.PATIENCE': 'learning_rate_n_to_compare',
    'TRAIN.LR.STOP': 'learning_rate_stop',
    'TRAIN.ADAM_EPSILON': 'adam_epsilon',
    'TRAIN.MAX_EPOCHS': 'target_num_epochs',
    'TRAIN.MAX_GRAD_NORM': 'max_grad_norm',
    'TRAIN.MODEL_DIR': 'lfads_save_dir',
    # 'TRAIN.OVERWRITE': False,
    'TRAIN.PATIENCE': 'n_epochs_early_stop',
    # 'TRAIN.TUNE_MODE': False,
    # 'TRAIN.USE_TB': False,
}

# load the default configuration dictionary (flattened)
flat_cfg_dict = flatten(yaml.full_load(open(cfg_path)))

# merge the samples with the config dictionary
flat_cfg_dict.update(cfg_samples)

class tuneValidation(tune.Trainable):
    def _setup(self, config):
        from lfads_tf2.models import LFADS # import here to avoid monopolizing GPU's

        self.config = config
        # ------ SETUP THE LFADS_TF2 MODEL ------
        def convert_cfg_for_lf2(cfg_dict):
            cfg_node = get_cfg_defaults() # get the LFADS defaults
            model_dir = path.join(self.logdir, 'lf2') # Use tune logdir as model_dir
            os.makedirs(model_dir)
            cfg_dict['TRAIN.MODEL_DIR'] = model_dir
            cfg_update = CN(unflatten(cfg_dict)) # update with samples and model_dir
            cfg_node.merge_from_other_cfg(cfg_update)
            return cfg_node        

        self.lf2_cfg = convert_cfg_for_lf2(config)
        self.lf2_model = LFADS(cfg_node=self.lf2_cfg)

        # ------ SETUP THE LFADSLITE MODEL ------
        def convert_cfg_for_lfl(cfg_node):
            cfg_dict = flatten(yaml.safe_load(cfg_node.dump())) # convert config node back to dict
            # convert HPs into the new format
            lfl_cfg = {convert[lf2_name]: val for lf2_name, val in cfg_dict.items() if lf2_name in convert}
            # add any HPs not used in lf2
            lfl_cfg.update({
                "kind": 'train',
                "keep_prob": 1-lfl_cfg['keep_prob'], # these are swapped from lfl to lf2
                "keep_ratio": 1-lfl_cfg['keep_ratio'],# these are swapped from lfl to lf2
                "cv_keep_ratio": 1-lfl_cfg['cv_keep_ratio'], # these are swapped from lfl to lf2
                "lfads_save_dir": path.join(self.logdir, 'lfl'), # save the lfl model in its own directory
                # LFL doesn't count epochs towards the target until after ramping
                # "target_num_epochs": lfl_cfg['target_num_epochs'] - \
                #     max(lfl_cfg['l2_increase_epochs'], lfl_cfg['kl_increase_epochs']), # set this to none when not using
                "target_num_epochs": None,
            })
            return lfl_cfg
        
        # convert the configuration to LFL JSON
        self.lfl_cfg = convert_cfg_for_lfl(self.lf2_cfg)
        self.lfl_run_script = path.join(validation_dir, 'lfl_train_and_sample.py')

        # ----- LOAD THE ANALYSIS DF FROM THE PREVIOUS EXPERIMENT -----
        if PREV_EXPERIMENT:
            self.prev_exp_df = prev_exp_df
    
    def _train(self):
        # ------ INITIALISE THE MODELS IDENTICALLY ------
        if SHARE_INIT:
            # create a random data matrix and pass through to initialize LF2 weights
            noise = np.random.randn(
                self.lf2_cfg.TRAIN.BATCH_SIZE, 
                self.lf2_cfg.MODEL.SEQ_LEN, 
                self.lf2_cfg.MODEL.DATA_DIM,
            ).astype(np.float32)
            _ = self.lf2_model(noise)
            # save the weights in the logdir
            weights_path = os.path.join(self.logdir, 'weights.h5')
            self.lf2_model.write_weights_to_h5(weights_path)

        results = {}
        # ------ TRAIN OR COPY THE LFADSLITE MODEL ------
        # set up LFL model training
        tfile = tempfile.NamedTemporaryFile()
        # save HPS to temporary JSON file
        with open(tfile.name, 'w') as f:
            json.dump(self.lfl_cfg, f)
        # run lfadslite from a python2 subprocess
        sh_str = f"/usr/bin/python {self.lfl_run_script} {tfile.name}"
        # throw away the lfadslite console output
        DEVNULL = open(os.devnull, 'wb')
        if not PREV_EXPERIMENT:
            # if previous experiment not specified, train a new LFL model
            # does not wait for completion before continuing - train in parallel
            proc = subprocess.Popen(sh_str.split(' '), stdout=DEVNULL, stderr=DEVNULL) 
        else:
            # look for a corresponding model in the previous experiment by merging dataframes
            cfg_dict = flatten(yaml.safe_load(self.lf2_cfg.dump())) # convert config node back to dict
            cfg_df = pd.DataFrame({'config/'+key: [val] for key, val in cfg_dict.items()})
            cfg_cols = [col for col in cfg_df.columns if 'MODEL_DIR' not in col and col in self.prev_exp_df.columns]
            # find the entry that matches the entire config
            match_df = self.prev_exp_df.merge(cfg_df, on=cfg_cols, suffixes=('_1', '_2')) 
            if len(match_df) == 0:
                # if a corresponding model is not found, train one
                print(f'Match not found in {PREV_EXPERIMENT}. Training LFL from scratch.')
                # subprocess.run will wait for the LFL model to finish, since we will only be using 0.5 GPU here.
                subprocess.run(sh_str.split(' '), stdout=DEVNULL, stderr=DEVNULL)
            else:
                if len(match_df) > 1:
                    print(f'More than one match found in {PREV_EXPERIMENT}. Selecting a random model to copy.')
                # get the previous logdir of the matching LFL model
                src = path.join(match_df.at[0, 'logdir'], 'lfl')
                dest = path.join(self.logdir, 'lfl')
                # copy the LFL model from the previous experiment to the current experiment
                print(f"Copying model with matching HPs from {src}")
                shutil.copytree(src, dest)

        DEVNULL.close()

        # ------ TRAIN THE LFADS_TF2 MODEL ------
        self.lf2_model.train() # model restores to lowest smoothed LVE
        if not PREV_EXPERIMENT:
            proc.wait() # wait for lfl training to finish before continuing
            tfile.close() # get rid of temporary file after lfadslite has seen it

        # ------ EVALUATE THE LFADSLITE MODEL ------
        # load the true rates
        train_truth, valid_truth = load_data(
            self.lf2_cfg.TRAIN.DATA.DIR, 
            prefix=self.lf2_cfg.TRAIN.DATA.PREFIX, 
            signal='truth'
        )[0]
        n_neurons = train_truth.shape[-1]
        # load the posterior sampled data and compute R2
        ps_files = sorted(glob(path.join(self.lfl_cfg['lfads_save_dir'], '*posterior_sample*')))
        with h5py.File(ps_files[0], mode='r') as h5file:
            train_rates = h5file['output_dist_params'][()]
        with h5py.File(ps_files[1], mode='r') as h5file:
            valid_rates = h5file['output_dist_params'][()]
        results['lfl_r2'] = r2_score(
            train_truth.reshape(-1, n_neurons), 
            train_rates.reshape(-1, n_neurons))
        results['lfl_val_r2'] = r2_score(
            valid_truth.reshape(-1, n_neurons), 
            valid_rates.reshape(-1, n_neurons))

        # ------ EVALUATE THE LFADS_TF2 MODEL ------
        # load the data, posterior sample from LF2, and compute R2
        self.lf2_model.sample_and_average()
        train_output, valid_output = load_posterior_averages(self.lf2_cfg.TRAIN.MODEL_DIR)
        train_rates, *_ = train_output
        valid_rates, *_ = valid_output
        results['lf2_r2'] = r2_score(
            train_truth.reshape(-1, n_neurons), 
            train_rates.reshape(-1, n_neurons))
        results['lf2_val_r2'] = r2_score(
            valid_truth.reshape(-1, n_neurons), 
            valid_rates.reshape(-1, n_neurons))

        # ------ LOAD THE TRAINING STATISTICS ------
        lf2_df = pd.read_csv(path.join(self.lf2_cfg.TRAIN.MODEL_DIR, 'train_data.csv'))
        lfl_df = pd.read_csv(
            path.join(self.lfl_cfg['lfads_save_dir'], 'fitlog_smoothed.csv'),
            names=['epoch', 'step', 'loss', 'val_loss', 'smth_nll_heldin', 'smth_nll_heldout', 
                   'smth_val_nll_heldin', 'smth_val_nll_heldout', 'wt_kl', 'val_wt_kl', 'wt_l2', 
                   'kl_wt', 'l2_wt', 'lr'],
            usecols=[1, 3, 5, 6, 8, 9, 10, 11, 19, 20, 22, 24, 26, 28],
        ).replace('NAN', np.nan)
        # get the model with the lowest smth_val_nll after ramping
        l2_last_ramp = self.lf2_model.cfg.TRAIN.L2.INCREASE_EPOCH + self.lf2_model.cfg.TRAIN.L2.START_EPOCH
        kl_last_ramp = self.lf2_model.cfg.TRAIN.KL.INCREASE_EPOCH + self.lf2_model.cfg.TRAIN.KL.START_EPOCH
        last_ramp = max([l2_last_ramp, kl_last_ramp])
        lf2_best_ix = lf2_df[lf2_df['epoch'] > last_ramp].smth_val_nll_heldin.idxmin()
        lfl_best_ix = lfl_df[lfl_df['epoch'] > last_ramp].smth_val_nll_heldin.idxmin()
        results.update({'lf2_'+key: val for key, val in lf2_df.iloc[lf2_best_ix].to_dict().items() if not np.isnan(val)})
        results.update({'lfl_'+key: val for key, val in lfl_df.iloc[lfl_best_ix].to_dict().items() if not np.isnan(val)})
        # include gradient norms
        gn = pd.read_csv(path.join(self.lfl_cfg['lfads_save_dir'], 'gradnorms.csv'), names=['gnorm']).replace('NAN', np.nan)
        results.update({'lfl_gnorm': gn.iloc[lfl_best_ix].gnorm})
        # include non-smoothed nll metrics
        non_smth_df = pd.read_csv(
            path.join(self.lfl_cfg['lfads_save_dir'], 'fitlog.csv'),
            names=['nll_heldin', 'nll_heldout', 'val_nll_heldin', 'val_nll_heldout'],
            usecols=[8, 9, 10, 11],
        ).replace('NAN', np.nan)
        results.update({'lfl_'+key: val for key, val in non_smth_df.iloc[lfl_best_ix].to_dict().items() if not np.isnan(val)})

        # ------ SIGNAL THAT TRAINING IS COMPLETE------
        results['training_iteration'] = 1

        return results


# ------ RUN THE VALIDATION ROUTINE ------
analysis = tune.run(
    tuneValidation,
    stop={'training_iteration': 1}, # only go through the training loop once
    name=EXPERIMENT_NAME,
    config=flat_cfg_dict,
    resources_per_trial=resources,
    num_samples=NUM_SAMPLES,
    sync_to_driver='# {source} {target}', # prevents rsync
    verbose=1,
)
