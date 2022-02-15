"""Runs a comparison of the inputs computed by LFL and LF2.

This script is intended to evaluate how well the two models 
are capturing information about the true inputs to the chaotic 
RNN. It calculates a linear mapping from LFL and LF2 rates to 
the true inputs, and then evaluates regression on validation 
inputs.

For LFL, we load from posterior sampling files, but for LF2 
we run posterior sampling in this script. For this reason, 
we distribute the work across the cluster using ray.
"""

import h5py
from os import path
from glob import glob
import matplotlib.pyplot as plt
import ray
from ray import tune
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from lfads_tf2.utils import load_data, load_posterior_averages
from lfads_tf2.models import LFADS

# ----- CONFIGURATION -----
exp_dir = '/snel/home/asedler/ray_results/validation_withCon_newConfig_full'
run_name = 'sampling_cd'
data_dir = '/snel/share/runs/PBT/paper-fix/lfadslite-testing/data_chaotic'
prefix = 'chaotic'
ray.init(address="localhost:6379") # connect to the cluster
# ray.init(local_mode=True) # debugging mode

# ----- LOAD THE TRUE INPUTS -----
h5_filename = sorted(glob(path.join(data_dir, prefix + '*')))[0]
with h5py.File(h5_filename, 'r') as h5file:
    # open the h5 file in a dictionary
    h5dict = {key: h5file[key][()] for key in h5file.keys()}

input_train_truth = h5dict['input_train_truth'].reshape(-1, 2)
input_valid_truth = h5dict['input_valid_truth'].reshape(-1, 2)

# ----- GET THE TRIAL FOLDERS -----
trial_dir_pattern = path.join(exp_dir, '*/')
trial_dirs = glob(trial_dir_pattern)

def posterior_sampling(config):
    trial_dir = config['trial_dir']

    # ----- LOAD THE LFADSLITE INPUTS -----
    train_ps_file_pattern = path.join(trial_dir, 'lfl/*train_posterior*')
    valid_ps_file_pattern = path.join(trial_dir, 'lfl/*valid_posterior*')
    train_ps_filename = glob(train_ps_file_pattern)[0]
    valid_ps_filename = glob(valid_ps_file_pattern)[0]
    with h5py.File(train_ps_filename, 'r') as h5file:
        train_co = h5file['controller_outputs'][()].reshape(-1, 10)
    with h5py.File(valid_ps_filename, 'r') as h5file:
        valid_co = h5file['controller_outputs'][()].reshape(-1, 10)

   # ----- PERFORM REGRESSION ON CO ----- 
    model = LinearRegression().fit(train_co, input_train_truth)
    lfl_val_r2 = r2_score(input_valid_truth, model.predict(valid_co))
    lfl_r2 = r2_score(input_train_truth, model.predict(train_co))
    print(f'LFL CO R2 -> TRAIN: {lfl_r2}, VALID: {lfl_val_r2}')

    # ----- LOAD THE ORIGINAL DATA -----
    train_data, valid_data = load_data(data_dir, prefix=prefix)[0]
    model_dir = path.join(trial_dir, 'lf2')

    # ----- PERFORM POSTERIOR SAMPLING
    try:
        model = LFADS(model_dir=model_dir)
        model.sample_and_average()
        train_output, valid_output = load_posterior_averages(model_dir)
        train_co = train_output.gen_inputs.reshape(-1, 10)
        valid_co = valid_output.gen_inputs.reshape(-1, 10)

        # ----- PERFORM REGRESSION ON CO -----
        model = LinearRegression().fit(train_co, input_train_truth)
        lf2_r2 = r2_score(input_train_truth, model.predict(train_co))
        lf2_val_r2 = r2_score(input_valid_truth, model.predict(valid_co))
    except AssertionError: # if not able to restore model, return arbitrary integer
        lf2_r2 = -999
        lf2_val_r2 = -999
    print(f'LF2 CO R2 -> TRAIN: {lf2_r2}, VALID: {lf2_val_r2}')
    
    tune.track.log(**{
        'lfl_r2': lfl_r2,
        'lf2_r2': lf2_r2,
        'lfl_val_r2': lfl_val_r2,
        'lf2_val_r2': lf2_val_r2,
    })

# run posterior sampling across the whole cluster
try:
    analysis = tune.run(
        posterior_sampling,
        name=run_name,
        config={'trial_dir': tune.grid_search(trial_dirs)},
        resources_per_trial={"cpu": 5, "gpu": 0.5},
        sync_to_driver='# {source} {target}',
        verbose=1,
    )
except ray.tune.error.TuneError:
    pass

exp_dir = path.join(path.expanduser('~/ray_results'), run_name)
df = tune.Analysis(exp_dir).dataframe()
# plot the R2 of decoding from controller outputs to true inputs
plt.scatter(df.lfl_val_r2, df.lf2_val_r2)
plt.plot([-0.1, 1.1], [-0.1, 1.1], c='k'); plt.xlim([-0.1, 0.4]); plt.ylim([-0.1, 0.4])
plt.xlabel('lfl_val_r2'); plt.ylabel('lf2_val_r2')
plt.title('decoding true inputs from controller outputs')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(path.join(exp_dir, 'input_decoding.png'))