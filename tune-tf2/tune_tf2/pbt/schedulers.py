import copy
import logging
import os
from collections import defaultdict, deque, namedtuple

import numpy as np
from ray.tune.schedulers import FIFOScheduler, TrialScheduler
from ray.tune.suggest.variant_generator import format_vars
from ray.tune.trial import Checkpoint, Trial
from tune_tf2.defaults import EXPLOIT_CSV, HPS_CSV, PBT_CSV
from tune_tf2.pbt import exploiters, explorers
from tune_tf2.pbt.hps import HyperParam

# use the ray logger
logger = logging.getLogger("ray.tune.schedulers.pbt")


TrialState = namedtuple(
    "TrialState",
    [
        "orig_tag",
        "score",
        "ckpt",
        "generation",
        "last_perturbation_time",
    ],
)


def make_experiment_tag(orig_tag, config, mutations):
    """Appends perturbed params to the trial name to show in the console."""

    resolved_vars = {}
    for k in mutations.keys():
        resolved_vars[("config", k)] = config[k]
    return "{}@perturbed[{}]".format(orig_tag, format_vars(resolved_vars))


class MultiStrategyPBT(FIFOScheduler):
    """A scheduler for Population Based Training.

    The main job of the scheduler is to decide which models
    and hyperparameter combinations will be run at each
    generation.
    """

    def __init__(
        self,
        hyperparam_space,
        exploit_method="binary_tournament",
        explore_method="perturb",
        time_attr="epoch",
        metric="smth_val_nll_heldin",
        mode="min",
        patience=4,
        max_generations=50,
        min_percent_improvement=0.0005,
    ):
        """Creates a MultiStrategyPBT scheduler.

        Parameters
        ----------
        hyperparam_space : dict of tune_tf2.pbt.HyperParam
            A dictionary mapping hyperparameter config names to
            HyperParam objects. It specifies allowable mutations
            for the hyperparameters.
        exploit_method : str, optional
            The method to use for exploitation, must be defined
            in tune_tf2.pbt.exploiters, by default "binary_tournament"
        explore_method : str, optional
            The method to use for exploration, must be defined
            in tune_tf2.pbt.explorers, by default "perturb"
        time_attr : str, optional
            The result attribute to use for tracking time,
            by default 'epoch'
        metric : str, optional
            The metric to optimize during PBT, by default
            "smth_val_nll_heldin"
        mode : {"min", "max"}, optional
            Whether to minimize or maximize the metric, by
            default "min"
        patience : int, optional
            The number of generations to use for determining if
            performance is still decreasing, by default 4
        max_generations : int, optional
            The maximum number of generations to train for, by default 50
        min_percent_improvement : float, optional
            The minimum percent improvement in metric per generation to
            allow training to continue, by default 0.0005
        """

        assert mode in ["min", "max"], "`mode` must be 'min' or 'max'!"

        def check_hp_space(space):
            for value in space.values():
                if isinstance(value, dict):
                    check_hp_space(value)
                elif not isinstance(value, HyperParam):
                    raise TypeError(
                        "`hyperparam_space` must be a hierarchical "
                        "dict of `HyperParam` objects."
                    )

        check_hp_space(hyperparam_space)

        FIFOScheduler.__init__(self)
        self._hyperparam_space = hyperparam_space
        self._time_attr = time_attr
        self._generation = 1
        self._max_generations = max_generations
        self._exploit_method = exploit_method
        self._explore_method = explore_method
        self._exploit = getattr(exploiters, exploit_method)
        self._explore = getattr(explorers, explore_method)
        self._metric = metric
        self._trial_state = defaultdict(list)
        self._trial_result = defaultdict(list)
        # best_scores is a circular buffer
        self._best_scores = deque(maxlen=patience)
        self._min_percent_improvement = min_percent_improvement
        self._percent_improvement = 0.0
        if mode == "max":
            self._metric_op = 1.0
        elif mode == "min":
            self._metric_op = -1.0
        self._num_perturbations = 0

    def on_trial_add(self, trial_runner, trial):
        """Called when a new trial is added to the trial runner."""
        trial_state = TrialState(
            orig_tag=trial.experiment_tag,
            score=None,
            ckpt=None,
            generation=0,
            last_perturbation_time=0,
        )
        self._trial_state[trial].append(trial_state)

    def on_trial_result(self, trial_runner, trial, result):
        """Called on each intermediate result returned by a trial.
        At this point, the trial scheduler can make a decision by returning
        one of CONTINUE, PAUSE, and STOP. This will only be called when the
        trial is in the RUNNING state."""

        prev_state = self._trial_state[trial][-1]
        time = result[self._time_attr]

        # save the state of this trial
        current_ckpt = trial_runner.trial_executor.save(
            trial, Checkpoint.MEMORY, result=result
        )
        current_state = TrialState(
            orig_tag=trial.experiment_tag,
            score=self._metric_op * result[self._metric],
            ckpt=current_ckpt,
            generation=prev_state.generation + 1,
            last_perturbation_time=time,
        )
        self._trial_state[trial].append(current_state)
        self._trial_result[trial].append(result)

        # wait for all of the other trials to finish
        all_trials = trial_runner.get_trials()
        other_trials = [t for t in all_trials if t != trial]
        for t in all_trials:
            state = self._trial_state[t][-1]
            # stop all of the trials if any of them is finished
            # TODO: fix this for early stopping.
            if t.status == Trial.TERMINATED:
                self._stop_trials(trial_runner, other_trials)
                return TrialScheduler.STOP

            if state.generation < self._generation:
                return TrialScheduler.PAUSE

        # record hyperparameters of this generation
        self._log_generation_config(all_trials)

        # stop everything if we have reached the final generation
        if self._generation >= self._max_generations:
            self._stop_trials(trial_runner, other_trials)
            return TrialScheduler.STOP

        # get the state of all trials for this generation
        def get_gen_state(t):
            return self._trial_state[t][self._generation]

        generation_state = {t: get_gen_state(t) for t in all_trials}
        # find the best metric for this generation and record in circular buffer
        best_score = max([s.score for s in generation_state.values()])
        self._best_scores.append(best_score)
        # check the percent improvement in the last `patience` generations.
        self._percent_improvement = (
            np.max(self._best_scores) - self._best_scores[0]
        ) / np.mean(np.abs(self._best_scores))

        # log the state of the PBT run
        pbt_state = {
            "generation": self._generation,
            "best_score": best_score,
            "percent_improvement": self._percent_improvement,
            "duration_sec": result["time_this_iter_s"],
            "epoch": result["epoch"],
        }
        self._log_pbt_state(trial.local_dir, pbt_state)
        # stop everything if the metric is not improving and buffer is full
        if (
            self._percent_improvement <= self._min_percent_improvement
            and len(self._best_scores) == self._best_scores.maxlen
        ):
            self._stop_trials(trial_runner, other_trials)
            return TrialScheduler.STOP

        # evolve if this is the last trial in the generation
        self._evolve_generation(trial_runner, trial, generation_state)
        self._generation += 1
        return TrialScheduler.CONTINUE

    def _stop_trials(self, trial_runner, trials):
        """ stops all trials in a list if they are not already terminated """
        for t in trials:
            if t.status != Trial.TERMINATED:
                trial_runner.trial_executor.stop_trial(t)

    def _log_generation_config(self, all_trials):
        """Saves the HP configuration of the generation to a CSV file."""
        gen_cfg_path = os.path.join(all_trials[0].local_dir, HPS_CSV)
        hp_names = sorted(self._hyperparam_space.keys())
        with open(gen_cfg_path, "a+") as gen_cfg_file:
            if os.stat(gen_cfg_path).st_size == 0:
                header = ["generation", "trial_id"] + hp_names
                gen_cfg_file.write(",".join(header) + "\n")
            for trial in all_trials:
                hp_values = [str(trial.config[name]) for name in hp_names]
                data = [str(self._generation), trial.trial_id] + hp_values
                gen_cfg_file.write(",".join(data) + "\n")

    def _log_pbt_state(self, local_dir, pbt_state):
        """Saves the state of PBT training to a CSV file"""
        pbt_state_path = os.path.join(local_dir, PBT_CSV)
        state_header = sorted(pbt_state.keys())
        with open(pbt_state_path, "a+") as pbt_state_file:
            if os.stat(pbt_state_path).st_size == 0:
                pbt_state_file.write(",".join(state_header) + "\n")
            data = [str(pbt_state[name]) for name in state_header]
            pbt_state_file.write(",".join(data) + "\n")

    def _log_exploit(self, old_state, new_state, old_trial, new_trial):
        """Keeps track of which models exploit which other models."""
        log_path = os.path.join(old_trial.local_dir, EXPLOIT_CSV)
        exploit_data = {
            "generation": self._generation,
            "old_trial": old_trial.trial_id,
            "new_trial": new_trial.trial_id,
            "old_score": old_state.score,
            "new_score": new_state.score,
        }
        header = sorted(exploit_data.keys())
        with open(log_path, "a+") as log_file:
            if os.stat(log_path).st_size == 0:
                log_file.write(",".join(header) + "\n")
            data = [str(exploit_data[name]) for name in header]
            log_file.write(",".join(data) + "\n")

    def _evolve_generation(self, trial_runner, last_trial, generation_state):
        """Generates the next set of trials."""

        trial_executor = trial_runner.trial_executor
        for trial in trial_runner.get_trials():
            trial_state = copy.deepcopy(generation_state[trial])
            # returns a trial to clone or None if the trial should persist
            trial_to_clone = self._exploit(trial_runner, trial, generation_state)
            if trial_to_clone is not None:
                # returns a modified config for the next generation
                new_state = copy.deepcopy(generation_state[trial_to_clone])
                new_config = self._explore(
                    trial_to_clone.config, self._hyperparam_space
                )
                logger.info(
                    "[exploit] transferring weights from trial "
                    "{} (score {:.5E}) -> {} (score {:.5E})".format(
                        trial_to_clone, new_state.score, trial, trial_state.score
                    )
                )

                self._log_exploit(trial_state, new_state, trial, trial_to_clone)

                new_tag = make_experiment_tag(
                    trial_state.orig_tag, new_config, self._hyperparam_space
                )

                reset_successful = trial_executor.reset_trial(
                    trial, new_config, new_tag
                )
                assert reset_successful, "Config transfer unsuccessful."
                # use the new state
                trial_state = new_state
                self._num_perturbations += 1

            # restart the trials using the appropriate checkpoints
            if trial == last_trial:
                trial_executor.restore(trial, trial_state.ckpt)
            else:
                trial_executor.start_trial(trial, trial_state.ckpt)

    def choose_trial_to_run(self, trial_runner):
        """Attempts to train all models to the same training iteration.
        This enables the PBT scheduler to support a greater number of
        concurrent trials than can fit in the cluster at any given time.
        """

        candidates = []
        for trial in trial_runner.get_trials():
            state = self._trial_state[trial][-1]
            if state.generation < self._generation and trial.status == Trial.PENDING:
                candidates.append(trial)

        return candidates[0] if candidates else None

    def debug_string(self):
        """Returns a human readable message for printing to the console."""

        pbt_mode = "Using PBT with `{}` explore and `{}` exploit. ".format(
            self._explore_method, self._exploit_method
        )
        pbt_state = "Generation {}/{}, {} Perturbs, {:.2E}% Improvement".format(
            self._generation,
            self._max_generations,
            self._num_perturbations,
            self._percent_improvement,
        )
        return pbt_mode + pbt_state
