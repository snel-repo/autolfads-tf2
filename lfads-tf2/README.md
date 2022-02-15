# lfads_tf2
A TensorFlow 2.0 Implementation of LFADS.

# Installation
```
# Create an environment for lfads_tf2
conda create --name lfads-tf2 python=3.7
# Install cudatoolkit and cudnn
conda install -c conda-forge cudatoolkit=10.0
conda install -c conda-forge cudnn=7.6
# Install lfads_tf2
cd lfads_tf2
pip install -e .
```

Example usage can be found in `train_lfads.py`

To build the HTML documentation, enter the `lfads_tf2/docs` directory with the `tf2-gpu` environment active and use the command `make html`. You can then open the HTML documentation at `lfads_tf2/docs/build/html` in your browser.
