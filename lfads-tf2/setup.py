from setuptools import find_packages, setup

setup(
    name="lfads_tf2",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow-gpu==2.0.0",
        "tensorflow-probability==0.8.0",
        "tensorflow-addons==0.6",
        "tensorboard==2.0.2",
        "yacs==0.1.6",
        "PyYAML>=5.1",
        "numpy==1.18.1",
        "pandas==1.*",
        "matplotlib==3.1.3",
        "scikit-learn==0.22.1",
        "gitpython==3.0.8",
        "h5py==2.10.0",
        "umap-learn==0.4.0",
        "sphinx",
        "sphinx_rtd_theme",
    ],
    author="Andrew Sedler",
    author_email="arsedler9@gmail.com",
    description="A Tensorflow 2.0 implementation of LFADS",
)
