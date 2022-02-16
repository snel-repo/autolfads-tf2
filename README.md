# AutoLFADS
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
