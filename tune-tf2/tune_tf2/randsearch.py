import logging
import os
import shutil
from glob import glob
from multiprocessing import Pool
from os import path
from pprint import pprint

import numpy as np
import pandas as pd
import ray
import yaml
from lfads_tf2.defaults import get_cfg_defaults
from lfads_tf2.utils import (
    flatten,
    read_fitlog,
    read_hps,
    restrict_gpu_usage,
    unflatten,
)
from ray import tune
from yacs.config import CfgNode as CN

logger = logging.getLogger(__name__)

MODEL_DIR_NAME = "model_dir"


def train_and_sample(config, checkpoint_dir=None):
    """Trains and performs posterior sampling on a single LFADS model

    Parameters
    ----------
    config : dict
        A dictionary of hyperparameters.
    checkpoint_dir : str, optional
        UNUSED
    """
    # Don't log the TensorFlow info messages on imports
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    # Relevant imports (Ray recommends doing it here)
    from lfads_tf2.models import LFADS

    # Use the logdir to store the model if TRAIN.MODEL_DIR is empty
    # (but allow it to be specified for reruns)
    if not config["TRAIN.MODEL_DIR"]:
        logdir = tune.get_trial_dir()
        model_dir = path.join(logdir, MODEL_DIR_NAME)
        config["TRAIN.MODEL_DIR"] = model_dir
    config["TRAIN.TUNE_MODE"] = True
    # Convert the config from dict to CfgNode
    cfg_update = CN(unflatten(config))
    # Merge the config with the defaults
    cfg_node = get_cfg_defaults()
    cfg_node.merge_from_other_cfg(cfg_update)
    # Create, train, and sample from the LFADS model
    model = LFADS(cfg_node=cfg_node)
    model.train()
    model.sample_and_average(batch_size=20)
    # TODO: Touch a file to signify that the run is complete


def gpu_restricted_train_and_sample(gpu, config):
    # Wrapper around train_and_sample that restricts to a given GPU
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    restrict_gpu_usage(gpu)
    train_and_sample(config)


class RandomSearch:
    def __init__(self, search_path):
        self.name = path.basename(search_path)
        self.local_dir = path.dirname(search_path)
        self.search_path = search_path
        if not path.exists(self.search_path):
            logger.info(f"Creating a new `RandomSearch` at {self.search_path}")
            self.complete = False
        else:
            # Get the model directories
            self.model_dirs = self.get_model_dirs()
            n_models = len(self.model_dirs)
            # Check that we found some model directories
            assert n_models > 0, "No models found."
            self.complete = True
            # Find the searched HPs
            hps_df = self.read_hps()
            diff_hps = hps_df.iloc[0] != hps_df.iloc[1]
            diff_hps = list(diff_hps[diff_hps].index)
            diff_hps.remove("TRAIN.MODEL_DIR")
            self.search_hps = diff_hps
            n_hps = len(self.search_hps)
            # Log info about the random search
            logger.info(
                f"Loading from existing `RandomSearch` of {n_models} "
                f"models and {n_hps} HPs at {self.search_path}"
            )

    def get_model_dirs(self):
        pattern = path.join(self.search_path, "*", MODEL_DIR_NAME)
        model_dirs = sorted(glob(pattern))

        return model_dirs

    def run(
        self,
        cfg_samples,
        train_func=train_and_sample,
        base_cfg_path=None,
        seed=0,
        num_samples=20,
        single_machine=True,
        resources_per_trial={"cpu": 3, "gpu": 0.5},
    ):

        if base_cfg_path is None:
            # Use the default hyperparameters
            logger.info("Using default base HPs.")
            flat_cfg_dict = flatten(get_cfg_defaults())
        else:
            # Load HPs from the config file as flattened dict
            flat_cfg_dict = flatten(yaml.full_load(open(base_cfg_path)))
        # Merge the samples with the config dictionary
        flat_cfg_dict.update(cfg_samples)
        # Connect to existing cluster or start on single machine
        address = None if single_machine else "localhost:10000"
        ray.init(address=address)
        # Run the random search
        np.random.seed(seed)
        analysis = tune.run(
            train_func,
            name=self.name,
            local_dir=self.local_dir,
            config=flat_cfg_dict,
            resources_per_trial=resources_per_trial,
            num_samples=num_samples,
            sync_to_driver=False,
            verbose=1,
        )
        self.model_dirs = self.get_model_dirs()
        self.search_hps = [
            h for h, r in cfg_samples.items() if type(r) is tune.sample.sample_from
        ]
        self.complete = True

        return analysis

    def read_fitlogs(self, processes=os.cpu_count()):
        assert self.complete, "Can't read fitlogs from an unfinished search."
        # Read the fitlog for each model
        with Pool(processes) as pool:
            fitlogs = pool.map(read_fitlog, self.model_dirs)
        # Assign arbitrary numbers to each trial, based on sorted directories
        for i, fitlog in enumerate(fitlogs):
            fitlog["trial_id"] = i
        # Combine all data into a single DataFrame
        randsearch_fitlog = pd.concat(fitlogs).reset_index(drop=True)

        return randsearch_fitlog

    def read_hps(self, processes=os.cpu_count()):
        assert self.complete, "Can't read HPs from an unfinished search."
        # Read the HPs for each model
        with Pool(processes) as pool:
            all_hps = pool.map(read_hps, self.model_dirs)
        # Flatten all HP dictionaries and create a DataFrame
        randsearch_hps = pd.DataFrame([flatten(hps) for hps in all_hps])

        return randsearch_hps

    def evaluate(self, eval_func, processes=os.cpu_count()):
        assert self.complete, "Can't read HPs from an unfinished search."
        # Run the evaluation function across all models
        with Pool(processes) as pool:
            results = pool.map(eval_func, self.model_dirs)
        results = pd.DataFrame(results)

        return results

    def retry_failed_runs(self, available_gpus=[0], models_per_gpu=2):

        # Check for failed runs
        failed_runs = []
        for model_dir in self.model_dirs:
            ckpt_folder = path.join(model_dir, "lfads_ckpts")
            # Failure indicated by missing checkpoints
            if not path.exists(ckpt_folder):
                failed_runs.append(model_dir)
        # Inform the user if no failed runs are found
        if len(failed_runs) == 0:
            logger.info(f"Found no failed runs at {self.search_path}")
            return
        # Give the user the opportunity to inspect the rerun directories
        logger.info("Found the following runs without checkpoints:")
        pprint(failed_runs)
        response = input("Move these runs to `failed_model` restart? [y/N]: ")
        if response.lower() != "y":
            return
        # Read the configuration files for these runs
        configs = [flatten(read_hps(d)) for d in failed_runs]
        # Move the failed models to a new home
        for model_dir in failed_runs:
            failed_model_dir = path.join(path.dirname(model_dir), "failed_model")
            shutil.move(model_dir, failed_model_dir)
        # Batch the models that can be run in parallel
        models_per_batch = len(available_gpus) * models_per_gpu
        n_batches = int(np.ceil(len(configs) / models_per_batch))
        # Run each batch on all of the available GPUs
        for i in range(n_batches):
            # Choose the number of configs that will fit on the machines
            end = min(len(configs), (i + 1) * models_per_batch)
            batch_configs = configs[i * models_per_batch : end]
            # Choose the GPUs that the models will be assigned to
            gpus = np.repeat(available_gpus, models_per_gpu)
            args = list(zip(gpus, batch_configs))
            # Train models using a parallel pool
            with Pool() as pool:
                # Starmap unpacks the tuples of arguments
                pool.starmap(gpu_restricted_train_and_sample, args)

    def retry_failed_sample(self, available_gpus=[0], models_per_gpu=2):

        # Check for runs without posterior sampling files
        failed_runs = []
        for model_dir in self.model_dirs:
            ps_file = path.join(model_dir, "posterior_samples.h5")
            # Failure indicated by missing posterior_samples.h5
            if not path.exists(ps_file):
                failed_runs.append(model_dir)
        # Inform the user if no failed runs are found
        if len(failed_runs) == 0:
            logger.info(f"Found no missing posterior samples at {self.search_path}")
            return
        # Give the user the opportunity to inspect the rerun directories
        logger.info("Found the following runs without posterior samples:")
        pprint(failed_runs)

        # TODO: Finish implementing posterior sampling rerun
