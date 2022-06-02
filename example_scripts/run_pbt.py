"""This is the template script for running PBT with lfads_tf2.
Please copy and modify this script to suit your needs.
"""

import shutil
from os import path

import ray
import yaml
from lfads_tf2.defaults import DEFAULT_CONFIG_DIR
from lfads_tf2.utils import flatten
from ray import tune
from tune_tf2.models import create_trainable_class
from tune_tf2.pbt.hps import HyperParam
from tune_tf2.pbt.schedulers import MultiStrategyPBT
from tune_tf2.pbt.trial_executor import SoftPauseExecutor

# ---------- PBT I/O CONFIGURATION ----------
# the default configuration file for the LFADS model
CFG_PATH = path.join(DEFAULT_CONFIG_DIR, "chaotic.yaml")
# the directory to save PBT runs (usually '~/ray_results')
PBT_HOME = path.expanduser("~/ray_results")
# the name of this PBT run (run will be stored at {PBT_HOME}/{PBT_NAME})
RUN_NAME = "pbt_test_run_2"  # the name of the PBT run
# the dataset to train the PBT model on
DATA_DIR = (
    "/snel/share/data/lf2_pbt_validation/chaotic/chaotic_05_replications/sample_0"
)
DATA_PREFIX = "chaotic"

# ---------- PBT RUN CONFIGURATION ----------
# whether to use single machine or cluster
SINGLE_MACHINE = False
# the number of workers to use - make sure machine can handle all
NUM_WORKERS = 40
# the resources to allocate per model
RESOURCES_PER_TRIAL = {"cpu": 2, "gpu": 0.5}
# the hyperparameter space to search
HYPERPARAM_SPACE = {
    "TRAIN.LR.INIT": HyperParam(
        1e-5, 5e-3, explore_wt=0.3, enforce_limits=True, init=0.004
    ),
    "MODEL.DROPOUT_RATE": HyperParam(
        0.0, 0.6, explore_wt=0.3, enforce_limits=True, sample_fn="uniform"
    ),
    "MODEL.CD_RATE": HyperParam(
        0.01, 0.7, explore_wt=0.3, enforce_limits=True, init=0.5, sample_fn="uniform"
    ),
    "TRAIN.L2.GEN_SCALE": HyperParam(1e-4, 1e-0, explore_wt=0.8),
    "TRAIN.L2.CON_SCALE": HyperParam(1e-4, 1e-0, explore_wt=0.8),
    "TRAIN.KL.CO_WEIGHT": HyperParam(1e-6, 1e-4, explore_wt=0.8),
    "TRAIN.KL.IC_WEIGHT": HyperParam(1e-5, 1e-3, explore_wt=0.8),
}
PBT_METRIC = "smth_val_nll_heldin"
EPOCHS_PER_GENERATION = 25
# ---------------------------------------------

# setup the data hyperparameters
dataset_info = {"TRAIN.DATA.DIR": DATA_DIR, "TRAIN.DATA.PREFIX": DATA_PREFIX}
# setup initialization of search hyperparameters
init_space = {name: tune.sample_from(hp.init) for name, hp in HYPERPARAM_SPACE.items()}
# load the configuration as a dictionary and update for this run
flat_cfg_dict = flatten(yaml.full_load(open(CFG_PATH)))
flat_cfg_dict.update(dataset_info)
flat_cfg_dict.update(init_space)
# Set the number of epochs per generation
tuneLFADS = create_trainable_class(EPOCHS_PER_GENERATION)
# connect to Ray cluster or start on single machine
address = None if SINGLE_MACHINE else "localhost:10000"
ray.init(address=address)
# create the PBT scheduler
scheduler = MultiStrategyPBT(HYPERPARAM_SPACE, metric=PBT_METRIC)
# Create the trial executor
executor = SoftPauseExecutor(reuse_actors=True)
# Create the command-line display table
reporter = tune.CLIReporter(metric_columns=["epoch", PBT_METRIC])
try:
    # run the tune job, excepting errors
    tune.run(
        tuneLFADS,
        name=RUN_NAME,
        local_dir=PBT_HOME,
        config=flat_cfg_dict,
        resources_per_trial=RESOURCES_PER_TRIAL,
        num_samples=NUM_WORKERS,
        sync_to_driver="# {source} {target}",  # prevents rsync
        scheduler=scheduler,
        progress_reporter=reporter,
        trial_executor=executor,
        verbose=1,
        reuse_actors=True,
    )
except tune.error.TuneError:
    pass

# load the results dataframe for this run
pbt_dir = path.join(PBT_HOME, RUN_NAME)
df = tune.Analysis(pbt_dir).dataframe()
df = df[df.logdir.apply(lambda path: "best_model" not in path)]
# find the best model
best_model_logdir = df.loc[df[PBT_METRIC].idxmin()].logdir
best_model_src = path.join(best_model_logdir, "model_dir")
# copy the best model somewhere easy to find
best_model_dest = path.join(pbt_dir, "best_model")
shutil.copytree(best_model_src, best_model_dest)
# perform posterior sampling
from lfads_tf2.models import LFADS  # noqa: E402

model = LFADS(model_dir=best_model_dest)
model.sample_and_average()
