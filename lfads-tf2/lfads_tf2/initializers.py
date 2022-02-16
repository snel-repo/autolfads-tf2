import numpy as np
from tensorflow.keras.initializers import Constant, RandomNormal, VarianceScaling


def ones_zeros(hidden_dim):
    """Creates a custom initializer for GRU cell bias.

    GRU bias initializer sets gate bias (upper 2/3 of matrix)
    to 1 and candidate bias (bottom 1/3 of matrix) to 0

    Parameters
    ----------
    hidden_dim : int
        The hidden dimension of the GRU Cell.

    Returns
    -------
    tf.keras.initializers.Constant
        The initial values of the GRU bias variable.
    """

    gate_init = np.ones(2 * hidden_dim)
    cand_init = np.zeros(hidden_dim)
    oz_np = np.concatenate([gate_init, cand_init], axis=0)
    return Constant(oz_np)


def make_variance_scaling(scale_dim):
    """Creates a variance-scaled initializer for a custom dimension.

    Parameters
    ----------
    scale_dim : int
        The dimension to use for scaling the variance of
        the initialization.

    Returns
    -------
    tf.keras.initializers.RandomNormal
        An initializer that uses values drawn from a distribution
        scaled according to `scale_dim`.
    """

    return RandomNormal(stddev=1 / np.sqrt(scale_dim))


# initializer from normal distribution, with variance scaled by 1 / sqrt(input_dim)
variance_scaling = VarianceScaling(mode="fan_in", distribution="untruncated_normal")
