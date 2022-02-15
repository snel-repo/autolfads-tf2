import yaml, shutil
import ray
from ray import tune
from os import path
from glob import glob

from tune_tf2.models import tuneLFADS
from tune_tf2.pbt.hps import HyperParam
from tune_tf2.pbt.schedulers import MultiStrategyPBT
from tune_tf2.pbt.trial_executor import SoftPauseExecutor
from tune_tf2.pbt.utils import read_pbt_fitlog
from tune_tf2.utils import early_stop, cleanup_pbt_dir
from lfads_tf2.utils import flatten

# ---------- PBT RUN CONFIGURATION -----------
CFG_NAME = 'chaotic.yaml' # the default configuration of the LFADS model
DATA_GROUP = 'chaotic/chaotic_05_replications' # the group ot datasets
NUM_WORKERS = 16 # should be 34 # the number of workers to use
# the hyperparameter space to search
HYPERPARAM_SPACE = {
    'TRAIN.LR.INIT': HyperParam(1e-5, 5e-3, explore_wt=0.3, 
        enforce_limits=True, init=0.004),
    'MODEL.DROPOUT_RATE': HyperParam(0.0, 0.6, explore_wt=0.3,
        enforce_limits=True, sample_fn='uniform'),
    'MODEL.CD_RATE': HyperParam(0.01, 0.7, explore_wt=0.3,
        enforce_limits=True, init=0.5, sample_fn='uniform'),
    'TRAIN.L2.GEN_SCALE': HyperParam(1e-4, 1e-0, explore_wt=0.8),
    'TRAIN.L2.CON_SCALE': HyperParam(1e-4, 1e-0, explore_wt=0.8),
    'TRAIN.KL.CO_WEIGHT': HyperParam(1e-6, 1e-4, explore_wt=0.8),
    'TRAIN.KL.IC_WEIGHT': HyperParam(1e-5, 1e-3, explore_wt=0.8),
}
PBT_METRIC = 'smth_val_nll_heldin'
MAX_GENERATIONS = 50
PBT_PATIENCE = 4
# ---------------------------------------------

# where to find the configuration file
cfg_path = path.join(
    path.expanduser('~/core/tune_tf2/config'), 
    CFG_NAME)
# the directory to store all of the PBT runs
PBT_HOME = path.join(
    path.expanduser('~/ray_results/pbt_validation_allImprovementsFixCD'), 
    DATA_GROUP)
# the directory from which to load the PBT datasets
DATA_HOME = path.join(
    '/snel/share/data/lf2_pbt_validation', 
    DATA_GROUP)

# update the config dictionary with initialization samples
init_space = {name: tune.sample_from(hp.init) for name, hp in HYPERPARAM_SPACE.items()}
flat_cfg_dict = flatten(yaml.full_load(open(cfg_path)))
flat_cfg_dict.update(init_space)

# connect to the cluster
# ray.init(address="localhost:6379")
ray.init()

# find all of the data directories
data_dirs = sorted(glob(path.join(DATA_HOME, 'sample_*')))
data_prefix = 'chaotic'

# perform a PBT run on each dataset
for data_dir in data_dirs:
    # use the data sample to name the run
    run_name = path.basename(data_dir)
    # use the new dataset
    flat_cfg_dict.update({
        'TRAIN.DATA.DIR': data_dir,
        'TRAIN.DATA.PREFIX': data_prefix})

    # create the PBT scheduler
    pbt = MultiStrategyPBT(
        HYPERPARAM_SPACE,
        exploit_method="binary_tournament",
        explore_method="perturb",
        time_attr='epoch',
        metric=PBT_METRIC,
        mode="min",
        patience=PBT_PATIENCE,
        max_generations=MAX_GENERATIONS,
        log_config=True)
    # Create the command-line table
    reporter = tune.CLIReporter(
        metric_columns=['epoch', PBT_METRIC])
    # Create the trial executor
    executor = SoftPauseExecutor(reuse_actors=True)

    # run the tune job
    try:
        analysis = tune.run(
            tuneLFADS,
            name=run_name,
            local_dir=PBT_HOME,
            stop={'epoch': 10000},
            config=flat_cfg_dict,
            resources_per_trial={"cpu": 2, "gpu": 0.5},
            num_samples=NUM_WORKERS,
            sync_to_driver='# {source} {target}', # prevents rsync
            scheduler=pbt,
            progress_reporter=reporter,
            trial_executor=executor,
            verbose=1,
            reuse_actors=True,
        )
    except tune.error.TuneError:
        pass

    # move the best worker somewhere easy to find
    pbt_dir = path.join(PBT_HOME, run_name)
    df = tune.Analysis(pbt_dir).dataframe()
    df = df[df.logdir.apply(lambda path: not 'best_model' in path)]
    # find the best model
    best_model_src = df.loc[df[PBT_METRIC].idxmin()].logdir
    # copy the best model somewhere it's easy to find
    best_model_dest = path.join(pbt_dir, 'best_model')
    shutil.copytree(best_model_src, best_model_dest)
    # update the model_dir so we can work with this model
    new_model_dir = path.join(best_model_dest, 'model_dir')
    best_cfg_path = path.join(new_model_dir, 'model_spec.yaml')
    best_cfg = yaml.full_load(open(best_cfg_path))
    best_cfg['TRAIN']['MODEL_DIR'] = new_model_dir
    with open(best_cfg_path, 'w') as best_model_spec:
        yaml.dump(best_cfg, best_model_spec)