import logging
import time
import traceback

from ray.tune.error import AbortTrialExecution
from ray.tune.ray_trial_executor import RayTrialExecutor
from ray.tune.trial import Trial

logger = logging.getLogger("ray.tune.ray_trial_executor")


class SoftPauseExecutor(RayTrialExecutor):
    """Pauses and restartstrials without removing them from the GPU."""

    def pause_trial(self, trial):
        """Pauses the trial."""
        trial_future = self._find_item(self._running, trial)
        if trial_future:
            self._paused[trial_future[0]] = trial
        assert trial.status == Trial.RUNNING, trial.status
        self.set_status(trial, Trial.PAUSED)

    def start_trial(self, trial, checkpoint=None):
        """Identical to `RayTrialExecutor.start_trial`, except does not
        allocate unless there is no `trial.runner`.
        """
        # --- custom ---
        if trial.runner is None:
            self._commit_resources(trial.resources)
        # --------------
        try:
            self._start_trial(trial, checkpoint)
        except AbortTrialExecution:
            logger.exception("Trial %s: Error starting runner, aborting!", trial)
            time.sleep(2)
            error_msg = traceback.format_exc()
            self._stop_trial(trial, error=True, error_msg=error_msg)
        except Exception:
            logger.exception("Trial %s: Unexpected error starting runner.", trial)
            time.sleep(2)
            error_msg = traceback.format_exc()
            self._stop_trial(trial, error=True, error_msg=error_msg)

    def _start_trial(self, trial, checkpoint=None, runner=None):
        """Identical to `RayTrialExecutor.start_trial`, except does not
        create a runner unless there is no `trial.runner`.
        """
        prior_status = trial.status
        # --- custom ---
        if trial.runner is None:
            if runner is None:
                reuse_allowed = checkpoint is not None or trial.has_checkpoint()
                runner = self._setup_remote_runner(trial, reuse_allowed)
            trial.set_runner(runner)
        # --------------
        self.restore(trial, checkpoint)
        self.set_status(trial, Trial.RUNNING)

        previous_run = self._find_item(self._paused, trial)
        if prior_status == Trial.PAUSED and previous_run:
            # If Trial was in flight when paused, self._paused stores result.
            self._paused.pop(previous_run[0])
            self._running[previous_run[0]] = trial
        elif not trial.is_restoring:
            self._train(trial)
