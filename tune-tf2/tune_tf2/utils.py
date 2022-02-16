import os
import shutil
from glob import glob

import numpy as np
from ray import tune


def uniform(min_bound, max_bound):
    """ returns a uniform sampling function for `hyperparam_mutations` """
    return lambda: float(np.random.uniform(min_bound, max_bound))


def loguniform(min_bound, max_bound, base=10):
    """ returns a log-uniform sampling function for `hyperparam_mutations` """
    logmin = np.log(min_bound) / np.log(base)
    logmax = np.log(max_bound) / np.log(base)
    return lambda: float(base ** (np.random.uniform(logmin, logmax)))


# --------------------- EARLY STOPPING FUNCTION ------------------------
BEST_VAL_LOSS = np.inf  # the best val_loss so far across all trials
BEST_TRIAL_ID = None  # the trial_id of the model with the best val_loss
CUR_PATIENCE = 0  # the number of generations since the best val_loss
PATIENCE = 5  # stop after 5 generations with no improvement to val_loss
STOP_ALL = False  # whether to stop all trials


def early_stop(trial_id, result):
    global BEST_VAL_LOSS, CUR_PATIENCE, STOP_ALL, BEST_TRIAL_ID

    # track patience for early stopping using only the best trial
    if result["val_loss"] < BEST_VAL_LOSS:
        BEST_VAL_LOSS = result["val_loss"]
        BEST_TRIAL_ID = trial_id
        CUR_PATIENCE = 0
    elif trial_id == BEST_TRIAL_ID:
        CUR_PATIENCE += 1
        # check if the patience has reached the threshold
        if CUR_PATIENCE >= PATIENCE:
            STOP_ALL = True
    # kill the model if it has a NaN metric
    has_nan = np.isnan(
        result["val_loss"]
    )  # TODO: THIS ISN'T QUITE RIGHT - SHOULD TELL IT TO EXPOIT, NOT STOP

    return STOP_ALL or has_nan


# ----------------------------------------------------------------------


def cleanup_pbt_dir(pbt_dir, best_model_dest=None):
    """Deletes old checkpoints and moves the best worker to `best_worker`

    Parameters
    ----------
    pbt_dir : str
        The directory of the PBT run.
    best_worker_dest : str (optional)
        The destination to copy the best model.
    """
    # delete the extra checkpoint files created by tune
    ckpt_folders = glob(os.path.join(pbt_dir, "*/checkpoint_*"))
    for folder in ckpt_folders:
        shutil.rmtree(folder)

    # get the analysis object and find the best model
    analysis_df = tune.Analysis(pbt_dir).dataframe()
    best_model_src = analysis_df.loc[analysis_df["val_loss"].idxmin()].logdir

    # copy the best model somewhere it's easy to find
    if best_model_dest is None:
        best_model_dest = os.path.join(pbt_dir, "best_model")
    shutil.copytree(best_model_src, best_model_dest)
