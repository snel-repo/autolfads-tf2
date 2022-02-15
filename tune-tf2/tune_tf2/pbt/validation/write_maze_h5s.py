import os
import re
from glob import glob
import h5py
import numpy as np
import scipy.signal
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

mat_dir = '/snel/share/share/derived/maze_data'
mat_paths = sorted(glob(os.path.join(mat_dir, '*.mat')))

output_dir = '/snel/share/share/derived/maze_data/spikes_rates_velocity_2ms'

DX = 0.001 # time between samples
BIN_SIZE = 2 # bin size in ms (or downsampling factor)
VAL_RATIO = 0.2 # percentage of the data to use for validation
LAMBDA = 1 # Tikhonov regularization penalty
DELAY = 90 # decoding delay in milliseconds

# store the R2 data in a dictionary
r2_data = {}

# for each of the matfiles given
for mat_path in mat_paths:
    base = os.path.basename(mat_path)
    ds_tag = os.path.splitext(base)[0] # get a tag for this dataset
    n_train_trials = int(int(re.findall(r'\d+', ds_tag)[0]) * (1-VAL_RATIO))
    r2_data[n_train_trials] = []
    # open the file
    with h5py.File(mat_path, 'r') as matfile:
        # iterate through each reference to a sample of trials
        sample_ref_array = matfile['data'][:,0]
        for sample_num, sample_ref in enumerate(sample_ref_array):
            sample_folder = os.path.join(output_dir, ds_tag, f'sample_{sample_num}')
            trials_ref = matfile[sample_ref][0,0]
            trials_group = matfile[trials_ref]

            def get_array_for_field(field):
                """ Collects and stacks the data for a given field """
                ref_array = trials_group[field][:,0]
                data_list = [matfile[ref][()] for ref in ref_array]
                return np.stack(data_list)

            # get the data for all of the trials in this sample
            spikes = get_array_for_field('y')
            positions = get_array_for_field('X')
            lfl_pbt_rates = get_array_for_field('rates')

            # downsample the spikes
            n_trials, n_timesteps, n_neurons = spikes.shape
            data = spikes[:, :-(n_timesteps % BIN_SIZE), :] # remove any extra time points
            data = data.transpose(0, 2, 1) # move the time to the end
            data = data.reshape(n_trials, n_neurons, -1, BIN_SIZE) # reshape adjacent bins to extra dim
            data = data.sum(axis=-1) # sum spikes in adjacent bins
            ds_spikes = data.transpose(0, 2, 1) # move time back to the middle

            # downsample the position
            n_trials, n_timesteps, _ = positions.shape
            data = positions[:, :-(n_timesteps % BIN_SIZE), :] # remove any extra time points
            ds_positions = scipy.signal.decimate(data, BIN_SIZE, axis=1) # downsample with antialiasing

            # convert position into velocity
            ds_velocity = np.gradient(ds_positions, DX, axis=1)

            # ensure that shapes are consistent
            assert lfl_pbt_rates.shape[:-1] == ds_spikes.shape[:-1] == ds_velocity.shape[:-1], \
                "Error: shape mismatch."

            # split into training and validation trials
            train_ixs = np.array([i for i in range(n_trials) if i % int(1/VAL_RATIO)])
            val_ixs = np.array([i for i in range(n_trials) if not i % int(1/VAL_RATIO)])

            # confirm that data is correct by decoding with shift of DELAY ms
            n_bins = DELAY // BIN_SIZE
            X = lfl_pbt_rates[train_ixs, :-n_bins, :].reshape(-1, n_neurons)
            X_val = lfl_pbt_rates[val_ixs, :-n_bins, :].reshape(-1, n_neurons)
            Y = ds_velocity[train_ixs, n_bins:, :].reshape(-1, 2)
            Y_val = ds_velocity[val_ixs, n_bins:, :].reshape(-1, 2)

            # # find the best regularized linear fit
            # clf = Ridge(alpha=LAMBDA)
            # clf.fit(X, Y)
            # Y_hat = clf.predict(X)
            # train_r2 = r2_score(Y, Y_hat)
            # Y_hat_val = clf.predict(X_val)
            # val_r2 = r2_score(Y_val, Y_hat_val)
            # print(f"sklearn - train: {train_r2}, valid: {val_r2}")

            # manually compute the best regularized linear fit
            X_bias = np.hstack([X, np.ones((len(X), 1))]) # add a bias feature
            W = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_bias.T, X_bias) + \
                LAMBDA * np.eye(n_neurons+1)), X_bias.T), Y)
            Y_hat = np.matmul(X_bias, W)
            train_r2 = r2_score(Y, Y_hat)
            X_bias_val = np.hstack([X_val, np.ones((len(X_val), 1))])
            Y_hat_val = np.matmul(X_bias_val, W)
            val_r2 = r2_score(Y_val, Y_hat_val)
            print(f"manual linear fit - train: {train_r2}, valid: {val_r2}")
            r2_data[n_train_trials].append(val_r2)

            # export the data to h5 files for LFADS
            os.makedirs(sample_folder, exist_ok=True)
            output_file = os.path.join(sample_folder, 'lfads_input.h5')
            with h5py.File(output_file, 'w') as hf:
                hf.create_dataset('train_inds', data=train_ixs)
                hf.create_dataset('valid_inds', data=val_ixs)
                hf.create_dataset('train_data', data=ds_spikes[train_ixs])
                hf.create_dataset('valid_data', data=ds_spikes[val_ixs])
                hf.create_dataset('train_vel', data=ds_velocity[train_ixs])
                hf.create_dataset('valid_vel', data=ds_velocity[val_ixs])
                hf.create_dataset('train_rates', data=lfl_pbt_rates[train_ixs])
                hf.create_dataset('valid_rates', data=lfl_pbt_rates[val_ixs])

            print(f"Wrote dataset to {output_file}.")

# plot the R2 to ensure data and decoding pipeline is correct
x, y = [], []
for n_train_trials, val_r2s in r2_data.items():
    x.extend([n_train_trials]*len(val_r2s))
    y.extend(val_r2s)
x, y = np.array(x), np.array(y)
plt.scatter(x + np.random.normal(scale=5, size=x.shape), y)
plt.title("Decoding Velocity from AutoLFADS Rates")
plt.xlabel('# of training trials')
plt.xlim(1800, 75)
plt.xscale('log')
plt.ylabel('Decoding R^2')
plt.ylim(0, 1)
    
fig_file = os.path.join(output_dir, 'val_r2.png')
plt.savefig(fig_file)
