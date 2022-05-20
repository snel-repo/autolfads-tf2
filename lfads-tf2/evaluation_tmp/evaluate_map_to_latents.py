import matplotlib.pyplot as plt
import pdb
import numpy as np
from sklearn.metrics import r2_score
from scipy.io import savemat
from os import path

from eval_utils import simple_ridge_wrapper
from lfads_tf2.utils import load_data, merge_chops, load_posterior_averages, merge_train_valid

run_dir = '/path/to/run/folder/'
data_dir = '/path/to/run/folder/lfads_input/'
model_dir = '/path/to/run/folder/model/'

truth_latents = load_data(
    data_dir, 
    prefix='lfads',
    merge_tv=True,
    signal='truth_latents')[0]
output = load_posterior_averages(model_dir, merge_tv=True)
lfads_rates, lfads_factors, *_ = output
lfads_output = {}
lfads_output['lfads_rates'] = lfads_rates
lfads_output['lfads_factors'] = lfads_factors
SAVE_DIR = path.join(run_dir, 'model_output', 'output.mat')
r2 = simple_ridge_wrapper(lfads_rates, truth_latents)
#savemat(SAVE_DIR, lfads_output)
pdb.set_trace()
