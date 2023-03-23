import logging
import os
from glob import glob
from os import path

import h5py
import numpy as np
import pandas as pd
import yaml
from lfads_tf2.tuples import SamplingOutput

logger = logging.getLogger(__name__)


def load_data(data_dir, prefix="", signal="data", merge_tv=False, with_filenames=False):
    """Loads the data from all data files in a given directory.

    Data should in the HDF5 file format and contain data arrays
    split into training and validation sets. Training data
    is indicated by prefixing the signal name with 'train_'
    and validation data is indicated by prefixing the signal
    name with 'valid_'.

    Parameters
    ----------
    data_dir : str
        The directory containing the data files.
    prefix : str, optional
        The prefix of the data files to be loaded from, by default ''
    signal : str, optional
        The name of data within each data file to load, by default 'data'
    merge_tv : bool, optional
        Whether to merge training and validation data, by default False
    with_filenames : bool, optional
        Whether to also return filenames, by default False

    Returns
    -------
    list of tuple of arrays or (list of tuple of arrays, list of str)
        A list of (train, valid) data pairs, one from each data file.
        Note that if `merge_tv` is `True`, a single array will be
        returned in the place of (train, valid) and

    Example
    -------
    If using this function on a single data file, be sure to select
    the first element of the list.

    >>> train_data, valid_data = load_data(data_dir, prefix)[0]

    """

    h5_filenames = sorted(glob(path.join(data_dir, prefix + "*")))
    datasets, filenames = [], []
    for h5_filename in h5_filenames:
        with h5py.File(h5_filename, "r") as h5file:
            # open the h5 file in a dictionary
            h5dict = {key: h5file[key][()] for key in h5file.keys()}

        # make sure this signal is found in the h5 file
        assert (
            "train_" + signal in h5dict
        ), f"Signal `{signal}` not found in {h5_filename}."

        # collect the data of interest
        train_data = h5dict["train_" + signal].astype(np.float32)
        valid_data = h5dict["valid_" + signal].astype(np.float32)

        if signal not in SamplingOutput._fields:
            # don't squeeze the output from LFADS
            train_data = train_data.squeeze()
            valid_data = valid_data.squeeze()
        if signal == "truth":
            # scale the true rates by the conversion factor
            cf = h5dict["conversion_factor"]
            train_data /= cf
            valid_data /= cf
        elif signal == "inds":
            # make smallest index zero for MATLAB compatibility
            min_ix = np.concatenate([train_data, valid_data]).min()
            train_data -= min_ix
            valid_data -= min_ix

        if merge_tv:
            # merge training and validation data
            if "train_inds" in h5dict and "valid_inds" in h5dict:
                # if there are index labels, use them to reassemble full data
                train_inds = h5dict["train_inds"]
                valid_inds = h5dict["valid_inds"]
                file_data = merge_train_valid(
                    train_data, valid_data, train_inds, valid_inds
                )
            else:
                logger.info(
                    "No indices found for merge. "
                    "Concatenating training and validation samples."
                )
                file_data = np.concatenate([train_data, valid_data], axis=0)
        else:
            file_data = train_data, valid_data

        # aggregate the data from all of the files
        datasets.append(file_data)
        if with_filenames:
            filenames.append(path.basename(h5_filename))

    # return all of the datasets
    if with_filenames:
        return datasets, filenames
    else:
        return datasets


def chop_data(data, overlap, window):
    """Rearranges an array of continuous data into overlapping segments.

    This low-level function takes a 2-D array of features measured
    continuously through time and breaks it up into a 3-D array of
    partially overlapping time segments.

    Parameters
    ----------
    data : np.ndarray
        A TxN numpy array of N features measured across T time points.
    overlap : int
        The number of points to overlap between subsequent segments.
    window : int
        The number of time points in each segment.

    Returns
    -------
    np.ndarray
        An SxTxN numpy array of S overlapping segments spanning
        T time points with N features.

    See Also
    --------
    lfads_tf2.utils.merge_chops : Performs the opposite of this operation.

    """

    if overlap>window/2:
        raise ValueError("Overlap must be less than half the window size.")

    shape = (
        int((data.shape[0] - window) / (window - overlap)) + 1,
        window,
        data.shape[-1],
    )
    strides = (
        data.strides[0] * (window - overlap),
        data.strides[0],
        data.strides[1],
    )
    chopped = (
        np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
        .copy()
        .astype("f")
    )
    return chopped


def merge_chops(data, overlap, orig_len=None, smooth_pwr=2):
    """Merges an array of overlapping segments back into continuous data.

    This low-level function takes a 3-D array of partially overlapping
    time segments and merges it back into a 2-D array of features measured
    continuously through time.

    Parameters
    ----------
    data : np.ndarray
        An SxTxN numpy array of S overlapping segments spanning
        T time points with N features.
    overlap : int
        The number of overlapping points between subsequent segments.
    orig_len : int, optional
        The original length of the continuous data, by default None
        will cause the length to depend on the input data.
    smooth_pwr : float, optional
        The power of smoothing. To keep only the ends of chops and
        discard the beginnings, use np.inf. To linearly blend the
        chops, use 1. Raising above 1 will increasingly prefer the
        ends of chops and lowering towards 0 will increasingly
        prefer the beginnings of chops (not recommended). To use
        only the beginnings of chops, use 0 (not recommended). By
        default, 2 slightly prefers the ends of segments.

    Returns
    -------
    np.ndarray
        A TxN numpy array of N features measured across T time points.

    See Also
    --------
    lfads_tf2.utils.chop_data : Performs the opposite of this operation.

    """

    if smooth_pwr < 1:
        logger.warning(
            "Using `smooth_pwr` < 1 for merging " "chops is not recommended."
        )

    merged = []
    full_weight_len = data.shape[1] - 2 * overlap

    assert full_weight_len>=0, "merge_chops: overlap cannot be larger than half of window size"

    # Create x-values for the ramp
    x = (
        np.linspace(1 / overlap, 1 - 1 / overlap, overlap)
        if overlap != 0
        else np.array([])
    )
    # Compute a power-function ramp to transition
    ramp = 1 - x ** smooth_pwr
    ramp = np.expand_dims(ramp, axis=-1)
    # Compute the indices to split up each chop
    split_ixs = np.cumsum([overlap, full_weight_len])
    for i in range(len(data)):
        # Split the chop into overlapping and non-overlapping
        first, middle, last = np.split(data[i], split_ixs)
        # Ramp each chop and combine it with the previous chop
        if i == 0:
            last = last * ramp
        elif i == len(data) - 1:
            first = first * (1 - ramp) + merged.pop(-1)
        else:
            first = first * (1 - ramp) + merged.pop(-1)
            last = last * ramp
        # Track all of the chops in a list
        merged.extend([first, middle, last])

    merged = np.concatenate(merged)
    # Indicate unmodeled data with NaNs
    if orig_len is not None and len(merged) < orig_len:
        nans = np.full((orig_len - len(merged), merged.shape[1]), np.nan)
        merged = np.concatenate([merged, nans])

    return merged


def merge_train_valid(train_data, valid_data, train_ixs, valid_ixs):
    """Merges training and validation numpy arrays using indices.

    This function merges training and validation numpy arrays
    in the appropriate order using arrays of their indices. The
    lengths of the indices must be the same as the first dimension
    of the corresponding data.

    Parameters
    ----------
    train_data : np.ndarray
        An N-dimensional numpy array of training data with
        first dimension T.
    valid_data : np.ndarray
        An N-dimensional numpy array of validation data with
        first dimension V.
    train_ixs : np.ndarray
        A 1-D numpy array of training indices with length T.
    valid_ixs : np.ndarray
        A 1-D numpy array of validation indices with length V.

    Returns
    -------
    np.ndarray
        An N-dimensional numpy array with dimension T + V.

    """

    if (
        train_data.shape[0] == train_ixs.shape[0]
        and valid_data.shape[0] == valid_ixs.shape[0]
    ):
        # if the indices match up, then we can use them to merge
        data = np.full_like(np.concatenate([train_data, valid_data]), np.nan)
        data[train_ixs.astype(int)] = train_data
        data[valid_ixs.astype(int)] = valid_data
    else:
        # if the indices do not match, train and
        # valid data may be the same (e.g. for priors)
        if np.allclose(train_data, valid_data, equal_nan=True):
            data = train_data
        else:
            raise ValueError(
                "shape mismatch: "
                f"Index shape {train_ixs.shape} does not "
                f"match the data shape {train_data.shape}."
            )
    return data


def flatten(dictionary, level=[]):
    """Flattens a dictionary by placing '.' between levels.

    This function flattens a hierarchical dictionary by placing '.'
    between keys at various levels to create a single key for each
    value. It is used internally for converting the configuration
    dictionary to more convenient formats. Implementation was
    inspired by `this StackOverflow post
    <https://stackoverflow.com/questions/6037503/python-unflatten-dict>`_.

    Parameters
    ----------
    dictionary : dict
        The hierarchical dictionary to be flattened.
    level : str, optional
        The string to append to the beginning of this dictionary,
        enabling recursive calls. By default, an empty string.

    Returns
    -------
    dict
        The flattened dictionary.

    See Also
    --------
    lfads_tf2.utils.unflatten : Performs the opposite of this operation.

    """

    tmp_dict = {}
    for key, val in dictionary.items():
        if type(val) == dict:
            tmp_dict.update(flatten(val, level + [key]))
        else:
            tmp_dict[".".join(level + [key])] = val
    return tmp_dict


def unflatten(dictionary):
    """Unflattens a dictionary by splitting keys at '.'s.

    This function unflattens a hierarchical dictionary by splitting
    its keys at '.'s. It is used internally for converting the
    configuration dictionary to more convenient formats. Implementation was
    inspired by `this StackOverflow post
    <https://stackoverflow.com/questions/6037503/python-unflatten-dict>`_.

    Parameters
    ----------
    dictionary : dict
        The flat dictionary to be unflattened.

    Returns
    -------
    dict
        The unflattened dictionary.

    See Also
    --------
    lfads_tf2.utils.flatten : Performs the opposite of this operation.

    """

    resultDict = dict()
    for key, value in dictionary.items():
        parts = key.split(".")
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return resultDict


def restrict_gpu_usage(gpu_ix=None):
    """Prevents TensorFlow from using all available GPU's.

    This function tells TensorFlow to allow GPU growth, which reduces
    GPU memory usage. Optionally, it also sets the CUDA_VISIBLE_DEVICES
    environment variable so that TF only uses the first visible GPU.

    Parameters
    ----------
    gpu_ix : int, optional
        The index of the GPU to use. Used to set CUDA_VISIBLE_DEVICES,
        by default None does not restrict to a single GPU

    """
    if gpu_ix is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ix)
    import tensorflow as tf

    # allow gpu growth
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def read_fitlog(model_dir, usecols=None, filename="train_data.csv"):
    fpath = path.join(model_dir, filename)
    fitlog = pd.read_csv(fpath, usecols=usecols)
    return fitlog


def read_hps(model_dir, flat=False, filename="model_spec.yaml"):
    fpath = path.join(model_dir, filename)
    with open(fpath, "r") as cfgfile:
        cfg = yaml.full_load(cfgfile)
    if flat:
        cfg = flatten(cfg)
    return cfg


def load_posterior_averages(
    model_dir, merge_tv=False, ps_filename="posterior_samples.h5"
):
    """Loads posterior sampling output from a file in the `model_dir`.

    This function is used for loading all of the posterior-sampled rates
    and other outputs from the HDF5 file created by LFADS posterior
    sampling.

    Parameters
    ----------
    model_dir : str
        The directory of the model to load from.
    merge_tv : bool, optional
        Whether to merge training and validation data, by default False.
    ps_filename : str, optional
        The name of the posterior sampling file to load from,
        by default 'posterior_samples.h5'.

    Returns
    -------
    lfads_tf2.tuples.SamplingOutput
        A namedtuple with properties corresponding to LFADS outputs. See
        fields of the tuple at `lfads_tf2.tuples.SamplingOutput` for more
        details.

    See Also
    --------
    lfads_tf2.models.LFADS.sample_and_average : Performs posterior sampling.

    """
    output_file = path.join(model_dir, ps_filename)
    prefix = path.splitext(ps_filename)[0]
    if not path.isfile(output_file):
        raise FileNotFoundError("No posterior sampling file found.")
    logger.info(f"Loading posterior samples from {output_file}")
    output = {}
    for field in SamplingOutput._fields:
        # load the posterior sampling data from the file
        output[field] = load_data(
            model_dir, prefix=prefix, signal=field, merge_tv=merge_tv
        )[0]
    # organize the posterior sampling data in namedtuples
    if merge_tv:
        sampling_output = SamplingOutput(**output)
    else:
        train_output = {field: pair[0] for field, pair in output.items()}
        valid_output = {field: pair[1] for field, pair in output.items()}
        sampling_output = (
            SamplingOutput(**train_output),
            SamplingOutput(**valid_output),
        )
    return sampling_output
