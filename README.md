# autolfads-tf2
A TensorFlow 2.0 implementation of LFADS and AutoLFADS.

# Installation
Clone the `autolfads-tf2` repo and create and activate a `conda` environment with Python 3.7. Use `conda` to install `cudatoolkit` and `cudnn` and `pip install` the `lfads_tf2` and `tune_tf2` packages with the `-e` (editable) flag. This will allow you to import these packages anywhere when your environment is activated, while also allowing you to edit the code directly in the repo.
```
git clone git@github.com:snel-repo/autolfads-tf2.git
cd autolfads-tf2
conda create --name autolfads-tf2 python=3.7
conda activate autolfads-tf2
conda install -c conda-forge cudatoolkit=10.0
conda install -c conda-forge cudnn=7.6
pip install -e lfads-tf2
pip install -e tune-tf2
```

# Usage

## Training single models with `lfads_tf2`
The first step to training an LFADS model is setting the hyperparameter (HP) values. All HPs, their descriptions, and their default values are given in the `defaults.py` module. Note that these default values are unlikely to work well on your dataset. To overwrite any or all default values, the user must define new values in a YAML file (example in `configs/lorenz.yaml`).

The `lfads_tf2.models.LFADS` constructor takes as input the path to the configuration file that overwrites default HP values. The path to the modeled dataset is also specified in the config, so `LFADS` will load the dataset automatically.

The `train` function will execute the training loop until the validation loss converges or some other stopping criteria is reached. During training, the model will save various outputs in the folder specified by `MODEL_DIR`. Console outputs will be saved to `train.log`, metrics will be saved to `train_data.csv`, and checkpoints will be saved in `lfads_ckpts`.

After training, the `sample_and_average` function can be used to compute firing rate estimates and other intermediate model outputs and save them to `posterior_samples.h5` in the `MODEL_DIR`.

We provide a simple example in `example_scripts/train_lfads.py`.

## Training AutoLFADS models with `tune_tf2`
The `autolfads-tf2` framework uses `ray.tune` to distribute models over a computing cluster, monitor model performance, and exploit high-performing models and their HPs.
### Setting up a `ray` cluster
If you'll be running AutoLFADS on a single machine, you can skip this section. If you'll be running across multiple machines, you must initialize the cluster using these instructions before you can submit jobs via the Python API.

Fill in the fields indicated by `<>`'s in the `ray_cluster_template.yaml`, and save this file somewhere accessible. Ensure that a range of ports is open for communication on all machines that you intend to use (e.g. `10000-10099` in the template). In your `autolfads-tf2` environment, start the cluster using `ray up <NEW_CLUSTER_CONFIG>`. The cluster may take up to a minute to get started. You can test that all machines are in the cluster by ensuring that all IP addresses are printed when running `example_scripts/ray_test.py`.
### Starting an AutoLFADS run
To run AutoLFADS, copy the `run_pbt.py` script and adjust paths and hyperparameters to your needs. Make sure to only use only as many workers as can fit on the machine(s) at once. If you want to run across multiple machines, make sure to set `SINGLE_MACHINE = False` in `run_pbt.py`. To start your PBT run, simply run `run_pbt.py`. When the run is complete, the best model will be copied to a `best_model` folder in your PBT run folder. The model will automatically be sampled and averaged and all outputs will be saved to `posterior_samples.h5`.
# References
Keshtkaran MR, Sedler AR, Chowdhury RH, Tandon R, Basrai D, Nguyen SL, Sohn H, Jazayeri M, Miller LE, Pandarinath C. A large-scale neural network training framework for generalized estimation of single-trial population dynamics. bioRxiv. 2021 Jan 1.

Keshtkaran MR, Pandarinath C. Enabling hyperparameter optimization in sequential autoencoders for spiking neural data. Advances in Neural Information Processing Systems. 2019; 32.
