"""
This script is both a demo for those who are new to running LFADS 
and a test script to rapidly test changes. The main steps to 
running LFADS are:

    1. Create a configuration YAML file to overwrite any or
        all of the defaults
    2. Create an LFADS object by passing the path to the 
        configuration file.
    3. Train the model using `model.train`
    4. Perform posterior sampling to create the posterior 
        sampling file using `model.sample_and_average`
    5. Load rates, etc. for further processing using 
        `lfads_tf2.utils.load_posterior_averages`

"""

from lfads_tf2.utils import restrict_gpu_usage
restrict_gpu_usage(gpu_ix=0)
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from os import path

from lfads_tf2.models import LFADS
from lfads_tf2.utils import load_data, merge_chops, load_posterior_averages
from lfads_tf2.defaults import get_cfg_defaults, DEFAULT_CONFIG_DIR

# create and train the LFADS model
cfg_path = path.join(DEFAULT_CONFIG_DIR, 'lorenz.yaml')
model = LFADS(cfg_path=cfg_path)
model.train()

# Read config to load data for evalution
cfg = get_cfg_defaults()
cfg.merge_from_file(cfg_path)
cfg.freeze()

# Load the spikes and the true rates
train_truth, valid_truth = load_data(
    cfg.TRAIN.DATA.DIR, 
    prefix=cfg.TRAIN.DATA.PREFIX, 
    signal='truth')[0]

# perform posterior sampling, then merge the chopped segments
model.sample_and_average()
model_dir = path.realpath(path.expanduser(cfg.TRAIN.MODEL_DIR))
train_output, valid_output = load_posterior_averages(model_dir)
train_lfads_rates, *_ = train_output
valid_lfads_rates, *_ = valid_output

# define how to compute r2
def compute_r2(truth, lfads_rates):
    n_trials, seg_len, _ = truth.shape
    truth_merged = merge_chops(truth, 0, n_trials * seg_len)
    lfads_rates_merged = merge_chops(lfads_rates, 0, n_trials * seg_len)
    r2 = r2_score(
        truth_merged, 
        lfads_rates_merged, 
        multioutput='uniform_average')
    return r2

# compute R2 for training and validation data
train_r2 = compute_r2(train_truth, train_lfads_rates)
valid_r2 = compute_r2(valid_truth, valid_lfads_rates)
print(f"Train R2: {train_r2}   Valid R2: {valid_r2}")

# plot and pause to allow the model to be examined
plt.plot(valid_lfads_rates[0,:,:])
fig_path = path.join(model_dir, 'rates.png')
plt.savefig(fig_path)
