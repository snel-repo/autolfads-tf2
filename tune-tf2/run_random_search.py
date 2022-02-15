import ray
import yaml
import numpy as np
from ray import tune
from os import path
from sklearn.metrics import r2_score
from yacs.config import CfgNode as CN

from lfads_tf2.defaults import get_cfg_defaults
from lfads_tf2.utils import (
    load_data, 
    load_posterior_averages,
    read_fitlog,
    flatten,
    unflatten,
)

# ========== SEARCH CONFIGURATION ==========
# User should set the constant values (variables w/ all caps)

# The parent directory of the random search
SEARCH_HOME = path.expanduser('~/tmp')
# The folder name of the random search
SEARCH_FOLDER = 'lorenz_randsearch'
# Set the absolute path to the config file (use relative path for demo)
relative_cfg_path = 'tune_tf2/config/lorenz.yaml'
CFG_PATH = path.join(
    path.dirname(path.abspath(__file__)),
    relative_cfg_path)
# Specify HPs whose initial values will be overwritten by samples
CFG_SAMPLES = {
    'MODEL.DROPOUT_RATE': tune.uniform(0.0, 0.6),
    'MODEL.CD_RATE': 0.5,
    'TRAIN.LR.INIT': 0.001,
    'TRAIN.KL.IC_WEIGHT': tune.loguniform(1e-6, 1e-3),
    'TRAIN.KL.CO_WEIGHT': tune.loguniform(1e-6, 1e-3),
    'TRAIN.L2.IC_ENC_SCALE': tune.loguniform(1e-5, 1e-3),
    'TRAIN.L2.CI_ENC_SCALE': tune.loguniform(1e-5, 1e-3),
    'TRAIN.L2.GEN_SCALE': tune.loguniform(1e-5, 1e0),
    'TRAIN.L2.CON_SCALE': tune.loguniform(1e-5, 1e0),
}
# Whether to use a single machine or connect to existing cluster
SINGLE_MACHINE = True
# How many models to train
NUM_MODELS = 20
# Whether to compute R2 with ground truth rates
TRUTH_R2 = True
# ==========================================

# Load HPs from the config file as flattened dict
flat_cfg_dict = flatten(yaml.full_load(open(CFG_PATH)))
# Merge the samples with the config dictionary
flat_cfg_dict.update(CFG_SAMPLES)
# Connect to existing cluster or start on single machine
address = None if SINGLE_MACHINE else 'localhost:10000'
ray.init(address=address)

# Define function for training, posterior sampling, and evaluation
def train_and_sample(config):
    # Relevant imports (Ray recommends doing it here)
    from lfads_tf2.models import LFADS
    # Convert the config from dict to CfgNode
    cfg_node = get_cfg_defaults()
    logdir = tune.get_trial_dir()
    model_dir = path.join(logdir, 'model_dir')
    config['TRAIN.MODEL_DIR'] = model_dir
    config['TRAIN.TUNE_MODE'] = True
    cfg_update = CN(unflatten(config))
    cfg_node.merge_from_other_cfg(cfg_update)
    # Create, train, and sample from the LFADS model
    model = LFADS(cfg_node=cfg_node)
    model.train()
    model.sample_and_average()
    # Find the best smoothed heldin NLL
    nll_metrics = {'train': 'smth_nll_heldin', 'valid': 'smth_val_nll_heldin'}
    fitlog = read_fitlog(model_dir, usecols=nll_metrics.values())
    best_epoch = fitlog[nll_metrics['valid']].idxmin()
    nll_results = {
        nll_metrics['valid']: fitlog.at[best_epoch, nll_metrics['valid']],
        nll_metrics['train']: fitlog.at[best_epoch, nll_metrics['train']],
    }
    tune.report(**nll_results)

    if TRUTH_R2:
        # Load the true rates from the original data file
        train_truth, valid_truth = load_data(
            config['TRAIN.DATA.DIR'],
            prefix=config['TRAIN.DATA.PREFIX'],
            signal='truth')[0]
        # Load the posterior-averaged rates from LFADS
        train_output, valid_output = load_posterior_averages(model_dir)
        train_rates, valid_rates = train_output.rates, valid_output.rates
        # Calculate coefficient of determination and report
        train_r2 = r2_score(train_truth.flatten(), train_rates.flatten())
        valid_r2 = r2_score(valid_truth.flatten(), valid_rates.flatten())
        tune.report(train_r2=train_r2, valid_r2=valid_r2)

# Run the random search
np.random.seed(0)
analysis = tune.run(
    train_and_sample,
    name=SEARCH_FOLDER,
    local_dir=SEARCH_HOME,
    config=flat_cfg_dict,
    resources_per_trial={"cpu": 3, "gpu": 0.5},
    num_samples=NUM_MODELS,
    sync_to_driver=False,
    verbose=1,
)
