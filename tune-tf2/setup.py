from setuptools import setup, find_packages

# NOTE: lfads_tf2 must be installed separately.

setup(
    name="tune_tf2",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "ray[tune]==0.8.7",
    ],
    author="Andrew Sedler",
    author_email="arsedler9@gmail.com",
    description="Hyperparameter utilities for LFADS",
)
