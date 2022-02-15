# tune_tf2
Hyperparameter tuning for LFADS in TensorFlow 2.0.

## Installation
Before installing `tune_tf2`, make sure `lfads_tf2` is installed in your environment. Then, clone and install the `tune_tf2` repository with the following commands:
```
git clone git@github.com:snel-repo/tune_tf2.git
cd tune_tf2
pip install -e .
```

## Running PBT
Once you have `tune_tf2` and its dependencies installed, update the `cluster.yaml` template with your username, a cluster identifier, and the conda environment in which you want to run LFADS. Copy the `run_pbt.py` script and adjust paths and hyperparameters to your needs. Make sure to only use only as many workers as can fit on the machine(s) at once. If you want to run across multiple machines, make sure to set `SINGLE_MACHINE = False` in `run_pbt.py` and ensure that ports 10000-10099 are open on all machines in the cluster. For multi-machine runs, the command `ray up cluster.yaml` must be executed in your shell to start up the cluster specified. To start your PBT run, simply run `run_pbt.py`. When the run is complete, the best model will be copied to a `best_model` folder in your PBT run folder. The model will automatically be sampled and averaged and all outputs will be saved to `posterior_samples.h5`.
