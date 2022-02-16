import copy
import random

import numpy as np


def perturb(config, hyperparam_space, min_perturb=0.002, clip=False, unbiased=True):
    """Perturbs a configuration based on the HPs specified.

    Parameters
    ----------
    config : dict
        The configuration to perturb.
    hyperparam_space : dict of tune_tf2.pbt.hps.HyperParam
        The hyperparameter space to search.
    min_perturb : float, optional
        The minimum perturbation allowed, by default 0.002
    clip : bool, optional
        Whether to clip the sampling output, by default True
        (False will resample to stay in HP boundaries)
    unbiased : bool, optional
        Whether to use unbiased sampling, by default False.
        Note that the default sampling method is biased to
        decline HPs

    Returns
    -------
    dict
        A `config` with mutated hyperparameters.
    """

    new_config = copy.deepcopy(config)
    for name, hp in hyperparam_space.items():
        # resample until new_value is within the bounds
        new_value = -np.inf
        # keep track of the number perturbation attempts
        num_tries = 0
        while not hp.min_bound <= new_value <= hp.max_bound:
            if unbiased:
                # sample in the log space
                min_perturb = np.log(1 + min_perturb * hp.explore_wt)
                max_perturb = np.log(1 + hp.explore_wt)
            else:
                # sample in the linear space
                min_perturb = min_perturb * hp.explore_wt
                max_perturb = hp.explore_wt
            # take a sample from the scaling space
            perturbation = random.choice(
                [
                    random.uniform(min_perturb, max_perturb),
                    random.uniform(-max_perturb, -min_perturb),
                ]
            )
            if unbiased:
                # convert back into the linear space
                scale = np.exp(perturbation)
            else:
                # center the perturbation at 1
                scale = 1 + perturbation
            # compute the new value candidate
            new_value = scale * new_config[name]
            # if something goes wrong with perturbation, clip
            num_tries += 1
            if clip or num_tries > 99:
                # clip this value to the boundaries
                new_value = np.clip(new_value, hp.min_bound, hp.max_bound)
        # assign the final value to the new config
        new_config[name] = float(new_value)

    return new_config
