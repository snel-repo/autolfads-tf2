import os
from os import path

from lfads_tf2.defaults import get_cfg_defaults
from lfads_tf2.tuples import BatchInput
from lfads_tf2.utils import unflatten
from ray import tune
from yacs.config import CfgNode as CN


def create_trainable_class(epochs_per_generation=50):
    """Creates a tuneLFADS class with specified number of epochs per
    generation. Uses static variable that can be accessed by instances.
    """
    # make epochs_per_generation global so it's visible inside class def
    global global_epg
    global_epg = epochs_per_generation

    class tuneLFADS(tune.Trainable):
        """A wrapper class that allows `tune` to interface with the
        LFADS model.
        """

        # Set default as static variable because class is initialized by tune
        epochs_per_generation = global_epg

        def setup(self, config):
            # Don't log the TensorFlow info messages on imports
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
            # Import inside the class so the run script doesn't take GPU memory
            import tensorflow as tf
            from lfads_tf2.models import LFADS

            # get the lfads config
            lfads_cfg = self.convert_tune_cfg(config)
            mcfg = lfads_cfg.MODEL
            # create the LFADS model and its checkpoint
            self.model = LFADS(cfg_node=lfads_cfg)
            # initialize the weights in the model by passing noise
            data_shape = (10, mcfg.SEQ_LEN, mcfg.DATA_DIM)
            output_seq_len = mcfg.SEQ_LEN - mcfg.IC_ENC_SEQ_LEN
            sv_mask_shape = (10, output_seq_len, mcfg.DATA_DIM)
            ext_input_shape = (10, output_seq_len, mcfg.EXT_INPUT_DIM)
            batch_of_noise = BatchInput(
                tf.random.uniform(shape=data_shape, dtype=tf.float32),
                tf.ones(shape=sv_mask_shape, dtype=tf.bool),
                tf.random.uniform(shape=ext_input_shape, dtype=tf.float32),
            )
            self.model.train_call(batch_of_noise)

        def step(self):
            # Use the static variable to determine the number of epochs
            num_epochs = self.epochs_per_generation
            # the first generation always completes ramping
            if self.model.cur_epoch < self.model.last_ramp_epoch:
                num_epochs += self.model.last_ramp_epoch
            # train for a fixed number of epochs
            for i in range(num_epochs):
                metrics = self.model.train_epoch()
            return metrics

        def save_checkpoint(self, ckpt_dir):
            # Only get model trainables so we overwrite config variables
            model_wts = [v.numpy() for v in self.model.trainable_variables]
            # Also need to include optimizer state
            optim_wts = self.model.optimizer.get_weights()
            checkpoint = {"model": model_wts, "optimizer": optim_wts}
            return checkpoint

        def load_checkpoint(self, checkpoint):
            # Transfer the trainable weights
            for v, array in zip(self.model.trainable_variables, checkpoint["model"]):
                v.assign(array)
            # Copy the optimizer momentum terms
            self.model.optimizer.set_weights(checkpoint["optimizer"])

        def reset_config(self, new_config):
            new_cfg_node = self.convert_tune_cfg(new_config)
            self.model.update_config({"cfg_node": new_cfg_node})
            return True

        def convert_tune_cfg(self, flat_cfg_dict):
            """Converts the tune config dictionary into a CfgNode for LFADS."""
            # get the LFADS defaults
            cfg_node = get_cfg_defaults()
            # Use the tune logging directory as the model_dir
            flat_cfg_dict["TRAIN.MODEL_DIR"] = path.join(self.logdir, "model_dir")
            flat_cfg_dict["TRAIN.TUNE_MODE"] = True
            # update the config with the samples and model directory
            cfg_update = CN(unflatten(flat_cfg_dict))
            cfg_node.merge_from_other_cfg(cfg_update)

            return cfg_node

    return tuneLFADS
