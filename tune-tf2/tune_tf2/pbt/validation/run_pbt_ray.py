import os
import yaml
import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
import numpy as np

from tune_tf2.models import tuneLFADS
from tune_tf2.utils import early_stop, cleanup_pbt_dir, uniform, loguniform
from lfads_tf2.utils import flatten


# specify the default HPs and architecture
cfg_path = '/snel/home/asedler/core/tune_tf2/validation/config/maze.yaml'

# specify HPs whose initial values will be overwritten by samples
cfg_samples = {
    'MODEL.DROPOUT_RATE': tune.uniform(0.0, 0.6),
    'MODEL.CD_RATE': 0.5,
    'TRAIN.LR.INIT': 0.001,
    'TRAIN.KL.IC_WEIGHT': tune.loguniform(1e-6, 1e-3),
    'TRAIN.KL.CO_WEIGHT': tune.loguniform(1e-6, 1e-3),
    'TRAIN.L2.IC_ENC_SCALE': tune.loguniform(1e-5, 1e-3),
    'TRAIN.L2.CI_ENC_SCALE': tune.loguniform(1e-5, 1e-3),
    'TRAIN.L2.GEN_SCALE': tune.loguniform(1e-5, 1e0),
    'TRAIN.L2.CON_SCALE': tune.loguniform(1e-5, 1e0),
}

# specify mutation ranges of HPs
hyperparam_mutations = {
    'MODEL.DROPOUT_RATE': uniform(0.0, 0.6),
    'MODEL.CD_RATE': uniform(0.1, 0.7),
    'TRAIN.LR.INIT': loguniform(1e-5, 1e-3),
    'TRAIN.KL.IC_WEIGHT': loguniform(1e-6, 1e-3),
    'TRAIN.KL.CO_WEIGHT': loguniform(1e-6, 1e-3),
    'TRAIN.L2.IC_ENC_SCALE': loguniform(1e-5, 1e-3),
    'TRAIN.L2.CI_ENC_SCALE': loguniform(1e-5, 1e-3),
    'TRAIN.L2.GEN_SCALE': loguniform(1e-5, 1e0),
    'TRAIN.L2.CON_SCALE': loguniform(1e-5, 1e0),
}

# load the default configuration dictionary (flattened)
flat_cfg_dict = flatten(yaml.full_load(open(cfg_path)))

# merge the samples with the config dictionary
flat_cfg_dict.update(cfg_samples)

# connect to the cluster
ray.init(address="170.140.140.223:43104")

# create the PBT scheduler
pbt_sched = PopulationBasedTraining(
    time_attr='epoch',
    metric='val_loss',
    mode='min',
    perturbation_interval=1,
    hyperparam_mutations=hyperparam_mutations,
    quantile_fraction=0.2,
    resample_probability=0.2,
) 

# run the tune job
run_name = 'pbt_mazetest'
try:
    analysis = tune.run(
        tuneLFADS,
        name=run_name,
        stop=early_stop,
        config=flat_cfg_dict,
        resources_per_trial={"cpu": 2, "gpu": 0.5},
        num_samples=20,
        sync_to_driver='# {source} {target}', # prevents rsync
        scheduler=pbt_sched,
        verbose=1,
        reuse_actors=True,
    )
except tune.error.TuneError:
    pass

# clean up the large checkpoint files and create `best_worker`
pbt_dir = os.path.join('~/ray_results', run_name)
cleanup_pbt_dir(pbt_dir)

import pdb; pdb.set_trace()