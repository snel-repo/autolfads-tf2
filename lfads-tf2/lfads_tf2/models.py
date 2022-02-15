from lfads_tf2.utils import restrict_gpu_usage
restrict_gpu_usage()
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import Progbar
from tensorflow.keras import Model, Sequential

from tensorflow_addons.layers import GroupNormalization
import tensorflow_probability as tfp
tfd = tfp.distributions

from sklearn.decomposition import PCA
from importlib import reload
import matplotlib.pyplot as plt
import logging.config
import numpy as np
import pandas as pd
import git, shutil, h5py, yaml, copy, sys, os, io

from lfads_tf2.defaults import get_cfg_defaults
from lfads_tf2.utils import load_data, flatten, load_posterior_averages
from lfads_tf2.tuples import LoadableData, BatchInput, DecoderInput, \
    LFADSInput, LFADSOutput, SamplingOutput
from lfads_tf2.layers import Encoder, Decoder, AutoregressiveMultivariateNormal
from lfads_tf2.initializers import variance_scaling
from lfads_tf2.regularizers import DynamicL2


class LFADS(Model):
    """
    Defines the LFADS model, a sequential autoencoder which takes as 
    input a batch of time segments of binned neural spikes, computes 
    a distributions over initial conditions for a generator network and
    a sequence of inputs to a controller network and then uses them 
    to generate a sequence of controller output posterior distributions, 
    generator states, factors, and estimated Poisson rates for the observed
    neurons. Note that `Decoder` is a subclass of `tensorflow.keras.Model`.
    """
    def __init__(self, cfg_node=None, cfg_path=None, model_dir=None):
        """ Initializes an LFADS object.

        This method will create a new model based on a specified configuration
        and create a `model_dir` in which to store all model-related data. 

        Parameters
        ----------
        cfg_node: yacs.config.CfgNode, optional
            yacs CfgNode to override the default CfgNode
        cfg_path: str, optional
            A path to a YAML update for the default YACS config node. 
            If not provided, use the defaults.
        model_dir: str, optional
            If provided, an LFADS run directory.
        """
        super(LFADS, self).__init__()

        # Check args
        num_set = sum([x != None for x in [cfg_node, cfg_path, model_dir]])
        if num_set > 1:
            raise ValueError(
                "Only one of `cfg_node`, `cfg_path`, or `model_dir` may be set")

        # Get the commit information for this lfads_tf2 code
        repo_path = os.path.realpath(
            os.path.join(os.path.dirname(__file__), '..'))
        repo = git.Repo(path=repo_path)
        git_data = {
            'commit': repo.head.object.hexsha,
            'modified': [diff.b_path for diff in repo.index.diff(None)],
            'untracked': repo.untracked_files,
        }

        if model_dir: # Load existing model - model_dir should contain its own cfg
            print("Loading model from {}.".format(model_dir))
            assert os.path.exists(model_dir), \
                "No model exists at {}".format(model_dir)
            ms_path = os.path.join(model_dir, 'model_spec.yaml')

            cfg = get_cfg_defaults()
            cfg.merge_from_file(ms_path)
            # check that the model directory is correct
            if not cfg.TRAIN.MODEL_DIR == model_dir:
                print("The `model_dir` in the config file doesn't match "
                "the true model directory. Updating and saving new config.")
                cfg.TRAIN.MODEL_DIR = model_dir
                ms_path = os.path.join(model_dir, 'model_spec.yaml')
                with open(ms_path, 'w') as ms_file:
                    ms_file.write(cfg.dump())
            cfg.freeze()

            # check that code used to train the model matches this code
            git_data_path = os.path.join(model_dir, 'git_data.yaml')
            trained_git_data = yaml.full_load(open(git_data_path))
            if trained_git_data != git_data:
                print(
                    'This `lfads_tf2` may not match the one '
                    'used to create the model.'
                )
            self.is_trained = True
            self.from_existing = True
        else: # Fresh model
            # Read config and prepare model directory
            print("Initializing new model.")
            cfg = get_cfg_defaults()
            if cfg_node:
                cfg.merge_from_other_cfg(cfg_node)
            elif cfg_path:
                cfg.merge_from_file(cfg_path)
            else:
                print("WARNING - Using default config")
            cfg.TRAIN.DATA.DIR = os.path.expanduser(cfg.TRAIN.DATA.DIR)
            cfg.TRAIN.MODEL_DIR = os.path.expanduser(cfg.TRAIN.MODEL_DIR)
            cfg.freeze()

            # Ensure that the model directory is handled appropriately
            model_dir = cfg.TRAIN.MODEL_DIR
            if not cfg.TRAIN.TUNE_MODE:
                if cfg.TRAIN.OVERWRITE:
                    if os.path.exists(model_dir):
                        print("Overwriting model directory...")
                        shutil.rmtree(model_dir)
                else:
                    assert not os.path.exists(model_dir), \
                        (f"A model already exists at `{model_dir}`. "
                        "Load it by explicitly providing the path, or specify a new `model_dir`.")
            # Create model directory
            os.makedirs(model_dir, exist_ok=True)
            
            # Save the model_spec
            try:
                ms_path = os.path.join(model_dir, 'model_spec.yaml')
                with open(ms_path, 'w') as ms_file:
                    ms_file.write(cfg.dump())
            except FileNotFoundError:
                print("File error. Please check that TUNE_MODE is set correctly.")
                raise

            # Save the git commit information
            git_data_path = os.path.join(model_dir, 'git_data.yaml')
            with open(git_data_path, 'w') as git_data_file:
                yaml.dump(git_data, git_data_file)

            # Create the model from the config file - must happen after model directory handling
            # because creation of the SummaryWriter automatically creates the model directory.
            self.is_trained = False
            self.from_existing = False


        # ==================================
        # ===== SET UP THE LFADS MODEL =====
        # ==================================

        # ===== SET UP THE LOGGERS =====
        # set up logging using the configuration file in this directory
        log_conf_path = os.path.join(os.path.dirname(__file__), 'logging_conf.yaml')
        logging_conf = yaml.full_load(open(log_conf_path))
        # in `TUNE_MODE`, restrict the console handler to WARNING's only
        if cfg.TRAIN.TUNE_MODE:
            logging_conf['handlers']['console']['level'] = 'WARNING'
        # tell the file handlers where to write output
        logging_conf['handlers']['logfile']['filename'] = os.path.join(cfg.TRAIN.MODEL_DIR, 'train.log')
        logging_conf['handlers']['csv']['filename'] = os.path.join(cfg.TRAIN.MODEL_DIR, 'train_data.csv')
        # set up logging from the configuation dict and create a global logger
        logging.config.dictConfig(logging_conf)
        self.lgr = logging.getLogger('lfads')
        self.csv_lgr = logging.getLogger('train_csv')
        # Prevent duplicate messages from another logging configuration
        self.lgr.propagate = False
        self.csv_lgr.propagate = False

        # ===== CHECK SEQ_LEN LOGIC =====
        # Make sure the total seq_len is longer than the IC seq_len
        assert cfg.MODEL.SEQ_LEN > cfg.MODEL.IC_ENC_SEQ_LEN
        cfg.defrost()
        # Make sure it's at least zero, reduces code complexity later
        cfg.MODEL.IC_ENC_SEQ_LEN = max(cfg.MODEL.IC_ENC_SEQ_LEN, 0)
        cfg.freeze()
        if cfg.MODEL.IC_ENC_SEQ_LEN > 0:
            self.lgr.info(f"Using the first {cfg.MODEL.IC_ENC_SEQ_LEN} steps "
                "to encode initial condition. Inferring rates for the "
                f"remaining {cfg.MODEL.SEQ_LEN - cfg.MODEL.IC_ENC_SEQ_LEN} steps.")

        # ===== DECISION TO USE CONTROLLER =====
        self.use_con = all([
            cfg.MODEL.CI_ENC_DIM > 0, 
            cfg.MODEL.CON_DIM > 0, 
            cfg.MODEL.CO_DIM > 0])
        if not self.use_con:
            self.lgr.info("A controller-related dim was set to zero. "
                "Turning off all controller-related HPs.")
            cfg.defrost()
            cfg.MODEL.CI_ENC_DIM = 0
            cfg.MODEL.CON_DIM = 0
            cfg.MODEL.CO_DIM = 0
            cfg.TRAIN.L2.CI_ENC_SCALE = 0.0
            cfg.TRAIN.L2.CON_SCALE = 0.0
            cfg.TRAIN.KL.CO_WEIGHT = 0.0
            cfg.freeze()

        # ===== DECISION TO USE RAMPING =====
        self.use_kl_ramping = any([
            cfg.TRAIN.KL.IC_WEIGHT, 
            cfg.TRAIN.KL.CO_WEIGHT])
        self.use_l2_ramping = any([
            cfg.TRAIN.L2.IC_ENC_SCALE, 
            cfg.TRAIN.L2.GEN_SCALE, 
            cfg.TRAIN.L2.CI_ENC_SCALE, 
            cfg.TRAIN.L2.CON_SCALE])
        cfg.defrost()
        if not self.use_kl_ramping:
            self.lgr.info("No KL weights found. Turning off KL ramping.")
            cfg.TRAIN.KL.START_EPOCH = 0
            cfg.TRAIN.KL.INCREASE_EPOCH = 0
        if not self.use_l2_ramping:
            self.lgr.info("No L2 weights found. Turning off L2 ramping.")
            cfg.TRAIN.L2.START_EPOCH = 0
            cfg.TRAIN.L2.INCREASE_EPOCH = 0
        cfg.freeze()

        # ===== INITIALIZE CONFIG VARIABLES =====
        # create variables for L2 and KL ramping weights
        self.kl_ramping_weight = tf.Variable(0.0, trainable=False)
        self.l2_ramping_weight = tf.Variable(0.0, trainable=False)
        # create variables for dropout rates
        self.dropout_rate = tf.Variable(cfg.MODEL.DROPOUT_RATE, trainable=False)
        self.cd_keep = tf.Variable(1 - cfg.MODEL.CD_RATE, trainable=False)
        self.cd_pass_rate = tf.Variable(cfg.MODEL.CD_PASS_RATE, trainable=False)
        self.sv_keep = tf.Variable(1 - cfg.MODEL.SV_RATE, trainable=False)
        # create variables for loss and gradient modfiers
        self.max_grad_norm = tf.Variable(cfg.TRAIN.MAX_GRAD_NORM, trainable=False)
        self.loss_scale = tf.Variable(cfg.TRAIN.LOSS_SCALE, trainable=False)
        # create variable for learning rate
        self.learning_rate = tf.Variable(cfg.TRAIN.LR.INIT, trainable=False)
        # create a variable that indicates whether the model is training
        self.training = tf.Variable(False, trainable=False)
        # TODO: Move the lowd_readin to a subclass
        # add low-dim readin matrix for alignment compatibility
        if cfg.MODEL.READIN_DIM > 0:
            self.lowd_readin = Sequential([
                Dense(cfg.MODEL.READIN_DIM,
                      kernel_initializer=variance_scaling,
                      kernel_regularizer=DynamicL2(
                          scale=cfg.TRAIN.L2.READIN_SCALE),
                      name='lowd_readin'), # TODO: rename 'lowd_linear'
                GroupNormalization(groups=1, axis=-1, epsilon=1e-12)
            ])
        # compute total recurrent size of the model for L2 regularization
        def compute_recurrent_size(cfg):
            t_cfg, m_cfg = cfg.TRAIN, cfg.MODEL
            recurrent_units_and_weights = [
                (m_cfg.IC_ENC_DIM, t_cfg.L2.IC_ENC_SCALE), 
                (m_cfg.IC_ENC_DIM, t_cfg.L2.IC_ENC_SCALE), 
                (m_cfg.CI_ENC_DIM, t_cfg.L2.CI_ENC_SCALE), 
                (m_cfg.CI_ENC_DIM, t_cfg.L2.CI_ENC_SCALE), 
                (m_cfg.GEN_DIM, t_cfg.L2.GEN_SCALE), 
                (m_cfg.CON_DIM, t_cfg.L2.CON_SCALE)]
            model_recurrent_size = 0
            for units, weight in recurrent_units_and_weights:
                if weight > 0:
                    model_recurrent_size += 3 * units**2
            return model_recurrent_size
        # total size of all recurrent kernels - note there are three gates to calculate
        self.model_recurrent_size = compute_recurrent_size(cfg)

        # ===== CREATE THE PRIORS =====
        with tf.name_scope('priors'):
            # create the IC prior variables
            self.ic_prior_mean = tf.Variable(
                tf.zeros(cfg.MODEL.IC_DIM), 
                name='ic_prior_mean')
            self.ic_prior_logvar = tf.Variable(
                tf.fill([cfg.MODEL.IC_DIM], 
                    tf.math.log(cfg.MODEL.IC_PRIOR_VAR)), 
                trainable=False, 
                name='ic_prior_logvar')
            # create the CO prior variables
            trainable_decoder = not cfg.TRAIN.ENCODERS_ONLY
            if cfg.MODEL.CO_AUTOREG_PRIOR:
                self.logtaus = tf.Variable(
                    tf.fill([cfg.MODEL.CO_DIM], 
                        tf.math.log(cfg.MODEL.CO_PRIOR_TAU)),
                    trainable=self.use_con & trainable_decoder,
                    name='logtaus')
                self.lognvars = tf.Variable(
                    tf.fill([cfg.MODEL.CO_DIM], 
                        tf.math.log(cfg.MODEL.CO_PRIOR_NVAR)),
                    trainable=self.use_con & trainable_decoder,
                    name='lognvars')
            else:
                self.co_prior_mean = tf.Variable(
                    tf.zeros([cfg.MODEL.CO_DIM]),
                    trainable=self.use_con & trainable_decoder,
                    name='co_prior_mean')
                self.co_prior_logvar = tf.Variable(
                    tf.fill([cfg.MODEL.CO_DIM],
                        tf.math.log(cfg.MODEL.CO_PRIOR_NVAR)),
                    trainable=False,
                    name='co_prior_logvar')
        if cfg.MODEL.CO_AUTOREG_PRIOR:
            # create the autoregressive prior distribution
            self.co_prior = AutoregressiveMultivariateNormal(
                self.logtaus, self.lognvars, cfg.MODEL.CO_DIM, name='co_prior')
        # create the KL weight variables
        self.kl_ic_weight = tf.Variable(
            cfg.TRAIN.KL.IC_WEIGHT, trainable=False, name='kl_ic_weight')
        self.kl_co_weight = tf.Variable(
            cfg.TRAIN.KL.CO_WEIGHT, trainable=False, name='kl_co_weight')

        # ===== CREATE THE ENCODER AND DECODER =====
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        if cfg.TRAIN.ENCODERS_ONLY:
            # Turn off decoder training
            self.lgr.warn('Training encoder only.')
            self.decoder.trainable = False

        # ===== CREATE THE OPTIMIZER =====
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            epsilon=cfg.TRAIN.ADAM_EPSILON)

        # ===== CREATE THE CHECKPOINTS =====
        ckpt_dir = os.path.join(cfg.TRAIN.MODEL_DIR, 'lfads_ckpts')
        # checkpointing for least validation error model
        self.lve_ckpt = tf.train.Checkpoint(model=self)
        lve_ckpt_dir = os.path.join(ckpt_dir, 'least_val_err')
        self.lve_manager = tf.train.CheckpointManager(
                self.lve_ckpt, directory=lve_ckpt_dir, 
                max_to_keep=1, checkpoint_name='lve-ckpt')
        # checkpointing for the most recent model
        self.mrc_ckpt = tf.train.Checkpoint(model=self)
        mrc_ckpt_dir = os.path.join(ckpt_dir, 'most_recent')
        self.mrc_manager = tf.train.CheckpointManager(
            self.mrc_ckpt, directory=mrc_ckpt_dir, 
            max_to_keep=1, checkpoint_name='mrc-ckpt')
        
        # ===== CREATE TensorBoard SUMMARY_WRITER =====
        if cfg.TRAIN.USE_TB:
            self.summary_writer = tf.summary.create_file_writer(cfg.TRAIN.MODEL_DIR)

        # ===== CREATE TRAINING DISTRIBUTIONS =====
        self.cd_input_dist = tfd.Bernoulli(probs=self.cd_keep, dtype=tf.bool)
        self.cd_pass_dist = tfd.Bernoulli(probs=self.cd_pass_rate, dtype=tf.bool)
        self.sv_input_dist = tfd.Bernoulli(probs=self.sv_keep, dtype=tf.bool)

        # keep track of when ramping ends
        kl_last_ramp_epoch = cfg.TRAIN.KL.START_EPOCH + cfg.TRAIN.KL.INCREASE_EPOCH
        l2_last_ramp_epoch = cfg.TRAIN.L2.START_EPOCH + cfg.TRAIN.L2.INCREASE_EPOCH
        self.last_ramp_epoch = max([l2_last_ramp_epoch, kl_last_ramp_epoch])

        # attach the config to LFADS
        cfg.freeze()
        self.cfg = cfg

        # wrap the training call with SV and CD
        self._build_wrapped_call()

        # =========================================
        # ===== END OF SETTING UP LFADS MODEL =====
        # =========================================

        # specify the metrics to be logged to CSV (and TensorBoard if chosen)
        self.logging_metrics = [
            'epoch', # epoch of training
            'step', # step of training
            'loss', # total loss on training data
            'nll_heldin', # negative log likelihood on seen training data
            'nll_heldout', # negative log likelihood on unseen training data
            'smth_nll_heldin', # smoothed negative log likelihood on seen training data
            'smth_nll_heldout', # smoothed negative log likelihood on unseen training data
            'wt_kl', # total weighted KL penalty on training data
            'wt_co_kl', # weighted KL penalty on controller output for training data
            'wt_ic_kl', # weighted KL penalty on initial conditions for training data
            'wt_l2', # weighted L2 penalty of the recurrent kernels
            'gnorm', # global norm of the gradient
            'lr', # learning rate
            'kl_wt', # percentage of the weighted KL penalty to applied
            'l2_wt', # percentage of the weighted L2 penalty being applied
            'val_loss', # total loss on validation data
            'val_nll_heldin', # negative log likelihood on seen validation data
            'val_nll_heldout', # negative log likelihood on unseen validation data
            'smth_val_nll_heldin', # smoothed negative log likelihood on seen validation data
            'smth_val_nll_heldout', # smoothed negative log likelihood on unseen validation data
            'val_wt_kl', # total weighted KL penalty on validation data
            'val_wt_co_kl', # weighted KL penalty on controller output for validation data
            'val_wt_ic_kl', # weighted KL penalty on initial conditions for validation data
            'val_wt_l2',
        ]

        # don't log the HPs by default because they are static without PBT
        cfg_dict = flatten(yaml.safe_load(cfg.dump()))
        self.logging_hps = sorted(cfg_dict.keys()) if cfg.TRAIN.LOG_HPS else []
        
        # create the CSV header if this is a new train_data.csv
        csv_log = logging_conf['handlers']['csv']['filename']
        if not self.from_existing and os.stat(csv_log).st_size == 0:
            self.csv_lgr.info(','.join(self.logging_metrics + self.logging_hps))
        
        # load the training dataframe
        train_data_path = os.path.join(self.cfg.TRAIN.MODEL_DIR, 'train_data.csv')
        self.train_df = pd.read_csv(train_data_path, index_col='epoch')

        # create the metrics for saving data during training
        self.all_metrics = {name: tf.keras.metrics.Mean() for name in self.logging_metrics}

        # load the datasets
        self.load_datasets_from_file(cfg.TRAIN.DATA.DIR, cfg.TRAIN.DATA.PREFIX)

        # init training params (here so they are consistent between training sessions)
        self.cur_epoch = tf.Variable(0, trainable=False)
        self.cur_step = tf.Variable(0, trainable=False)
        self.cur_patience = tf.Variable(0, trainable=False)
        self.train_status = '<TRAINING>'
        self.prev_results = {}

        # tracking variables for early stopping
        if self.last_ramp_epoch > self.cur_epoch:
            self.train_status = '<RAMPING>'
        # build the graphs for forward pass and for training
        self.build_graph()
        # restore the LVE model
        if self.from_existing:
            self.restore_weights()
    
    def build_graph(self):
        # ===== AUTOGRAPH FUNCTIONS =====
        # compile the `_step` function into a graph for better speed
        mcfg = self.cfg.MODEL
        output_seq_len = mcfg.SEQ_LEN - mcfg.IC_ENC_SEQ_LEN
        data_shape = [None, mcfg.SEQ_LEN, mcfg.DATA_DIM]
        sv_mask_shape = [None, output_seq_len, mcfg.DATA_DIM]
        ext_input_shape = [None, output_seq_len, mcfg.EXT_INPUT_DIM]
        # single step of training or validation
        self._graph_step = tf.function(
            func=self._step,
            input_signature=[
                BatchInput(
                    tf.TensorSpec(shape=data_shape), 
                    tf.TensorSpec(shape=sv_mask_shape, dtype=tf.bool), 
                    tf.TensorSpec(shape=ext_input_shape))])
        # forward pass of LFADS
        self.graph_call = tf.function(
            func=self.call,
            input_signature=[
                LFADSInput(
                    tf.TensorSpec(shape=data_shape), 
                    tf.TensorSpec(shape=ext_input_shape)),
                tf.TensorSpec(shape=[], dtype=tf.bool)])

    def get_config(self):
        """ Get the entire configuration for this LFADS model.

        See the TensorFlow documentation for an explanation of serialization: 
        https://www.tensorflow.org/guide/keras/save_and_serialize#custom_objects

        Returns
        -------
        dict
            A dictionary containing the configuration node.
        """
        return {'cfg_node': self.cfg}

    @classmethod
    def from_config(cls, config):
        """ Initialize an LFADS model from this config.

        See the TensorFlow documentation for an explanation of serialization: 
        https://www.tensorflow.org/guide/keras/save_and_serialize#custom_objects
        
        Returns
        -------
        lfads_tf2.models.LFADS
            An LFADS model from this config node.
        """
        return cls(cfg_node=config['cfg_node'])

    def update_config(self, config):
        """Updates the configuration of the entire LFADS model.
        
        Updates configuration variables of the model. 
        Primarily used for updating hyperparameters during PBT.

        Parameters
        ----------
        config : dict
            A dictionary containing the new configuration node.

        """
        node = config['cfg_node']
        old_cfg = self.cfg.clone()
        # keep the original model directory
        node.defrost()
        node.TRAIN.MODEL_DIR = old_cfg.TRAIN.MODEL_DIR
        node.freeze()
        # Set the new config
        self.cfg = node
        # Update decoder training appropriately
        encoders_only = node.TRAIN.ENCODERS_ONLY
        self.decoder.trainable = not encoders_only
        if encoders_only ^ old_cfg.TRAIN.ENCODERS_ONLY:
            # If we are swapping ENCODERS_ONLY, reset the optimizer momentum
            # and retrace the graph (learning rate updated below)
            self.lgr.warn(f'`TRAIN.ENCODERS_ONLY` flipped to {encoders_only}. ' \
                'Resetting the optimizer and retracing the graph.')
            for var in self.optimizer.variables():
                var.assign(tf.zeros_like(var))
            self.build_graph()
        self.learning_rate.assign(node.TRAIN.LR.INIT)
        self.dropout_rate.assign(node.MODEL.DROPOUT_RATE)
        self.cd_keep.assign(1 - node.MODEL.CD_RATE)
        self.cd_pass_rate.assign(node.MODEL.CD_PASS_RATE)
        self.sv_keep.assign(1 - node.MODEL.SV_RATE)
        self.max_grad_norm.assign(node.TRAIN.MAX_GRAD_NORM)
        self.loss_scale.assign(node.TRAIN.LOSS_SCALE)
        self.kl_ic_weight.assign(node.TRAIN.KL.IC_WEIGHT)
        self.kl_co_weight.assign(node.TRAIN.KL.CO_WEIGHT)
        self.encoder.update_config(config)
        self.decoder.update_config(config)
        # Reset previous training results
        self.train_df = self.train_df.iloc[0:0]
        self.cur_patience.assign(0)
        self.prev_results = {}
        # Overwrite the configuration file
        # TODO: Add tracking of past configs and record cfg index in train_df
        ms_path = os.path.join(node.TRAIN.MODEL_DIR, 'model_spec.yaml')
        with open(ms_path, 'w') as ms_file:
            ms_file.write(node.dump())

    def _update_metrics(self, metric_values, batch_size=1):
        """Updates the `self.all_metrics` dictionary with a new batch of values.
        
        Parameters
        ----------
        metric_values : dict
            A dict of metric updates as key-value pairs. Contains only keys 
            found in `self.all_metrics`.
        batch_size : int, optional
            The length of the batch of data used to calculate this metric. 
            Defaults to a weight of 1 for metrics that do not involve the 
            size of the batch.
        """
        for name, value in metric_values.items():
            # weight each metric observation by size of the corresponding batch
            self.all_metrics[name].update_state(value, sample_weight=batch_size)

    def call(self, lfads_input, use_logrates=tf.constant(False)):
        """ Performs the forward pass on the LFADS object using one sample 
        from the posteriors. The graph mode version of this function 
        (`self.graph_call`) should be used when speed is preferred.

        Parameters
        ----------
        lfads_input : lfads_tf2.tuples.LFADSInput
            A namedtuple of tensors containing the data and external inputs.
        use_logrates : bool, optional
            Whether to return logrates, which are helpful for numerical 
            stability of loss during training, by default False.

        Returns
        -------
        lfads_tf2.tuples.LFADSOutput
            A namedtuple of tensors containing rates and posteriors.
        """
        mcfg = self.cfg.MODEL
        # separate the inputs
        data, ext_input = tf.nest.flatten(lfads_input)
        # pass data through low-dim readin for alignment compatibility
        if mcfg.READIN_DIM > 0 and not mcfg.ALIGN_MODE:
            data = self.lowd_readin(data)
        # encode spikes into generator IC distributions and controller inputs
        ic_mean, ic_stddev, ci = self.encoder(data, training=self.training)
        # generate initial conditions
        if mcfg.SAMPLE_POSTERIORS:
            # sample from the distribution of initial conditions
            ic_post = tfd.MultivariateNormalDiag(ic_mean, ic_stddev)
            ic = ic_post.sample()
        else:
            # pass mean in deterministic mode
            ic = ic_mean
        # pass initial condition and controller input through decoder network
        dec_input = DecoderInput(
            ic_samp=ic,
            ci=ci,
            ext_input=ext_input)
        dec_output = self.decoder(
            dec_input,
            training=self.training, 
            use_logrates=use_logrates
        )
        rates, co_mean, co_stddev, factors, gen_states, \
            gen_init, gen_inputs, con_states = dec_output

        return LFADSOutput(
            rates=rates, 
            ic_means=ic_mean, 
            ic_stddevs=ic_stddev, 
            co_means=co_mean, 
            co_stddevs=co_stddev,
            factors=factors,
            gen_states=gen_states,
            gen_init=gen_init,
            gen_inputs=gen_inputs,
            con_states=con_states)

    def load_datasets_from_file(self, data_dir, prefix):
        """A wrapper that loads LFADS datasets from a file.
        
        This function is used for loading datasets into LFADS, 
        including datasets that LFADS was not trained on. LFADS 
        functions like `train` or `sample_and_average` will use 
        whatever data has been loaded by this function. This 
        function is a wrapper around `load_datasets_from_arrays`.
        
        Note
        ----
        This function currently only loads data from one 
        file at a time.

        Parameters
        ----------
        data_dir : str
            The directory containing the data files.
        prefix : str, optional
            The prefix of the data files to be loaded from, 
            by default ''

        See Also
        --------
        lfads_tf2.models.LFADS.load_datasets_from_arrays : 
            Creates tf.data.Dataset objects to use with LFADS.

        """

        self.lgr.info(f"Loading datasets with prefix {prefix} from {data_dir}")
        # load the spiking data from the data file
        train_data, valid_data = load_data(data_dir, prefix=prefix)[0]
        # Load any external inputs
        if self.cfg.MODEL.EXT_INPUT_DIM > 0:
            train_ext, valid_ext = load_data(
                data_dir, prefix=prefix, signal='ext_input')[0]
        else:
            train_ext, valid_ext = None, None
        # Load any training and validation indices
        try:
            train_inds, valid_inds = load_data(
                data_dir, prefix=prefix, signal='inds')[0]
        except AssertionError:
            train_inds, valid_inds = None, None

        # create the dataset objects
        loadable_data = LoadableData(
            train_data=train_data,
            valid_data=valid_data,
            train_ext_input=train_ext,
            valid_ext_input=valid_ext,
            train_inds=train_inds,
            valid_inds=valid_inds)
        self.load_datasets_from_arrays(loadable_data)

    def load_datasets_from_arrays(self, loadable_data):
        """Creates TF datasets and attaches them to the LFADS object.

        This function builds dataset objects from input arrays.
        The datasets are used for shuffling, batching, data 
        augmentation, and more. These datasets are used by 
        both posterior sampling and training functions.

        Parameters
        ----------
        loadable_data : lfads_tf2.tuples.LoadableData
            A namedtuple containing the input data and external
            inputs for the training and validation sets. See 
            definition of LoadableData for more detail.

        See Also
        --------
        lfads_tf2.models.LFADS.load_datasets_from_file : 
            A wrapper around this function that loads from a file.

        """

        train_data, valid_data, train_ext, valid_ext, \
            train_inds, valid_inds = loadable_data

        if train_ext is None or valid_ext is None:
            # use empty tensors if there are no inputs
            train_ext = tf.zeros(train_data.shape[:-1] + (0,))
            valid_ext = tf.zeros(valid_data.shape[:-1] + (0,))
        # remove external inputs during the ic encoder segment
        ic_enc_seq_len = self.cfg.MODEL.IC_ENC_SEQ_LEN
        train_ext = train_ext[:, ic_enc_seq_len:, :]
        valid_ext = valid_ext[:, ic_enc_seq_len:, :]
        # identify maskable data based on sequence lengths
        sv_maskable_train_data = train_data[:, ic_enc_seq_len:, :]
        sv_maskable_valid_data = valid_data[:, ic_enc_seq_len:, :]
        # create the sample validation masks
        sv_seed = self.cfg.MODEL.SV_SEED
        train_sv_mask = self.sv_input_dist.sample(
            sample_shape=tf.shape(sv_maskable_train_data), seed=sv_seed)
        valid_sv_mask = self.sv_input_dist.sample(
            sample_shape=tf.shape(sv_maskable_valid_data), seed=sv_seed)
        # package up the data into tuples and use to build datasets
        self.train_tuple = BatchInput(
            data=train_data, 
            sv_mask=train_sv_mask, 
            ext_input=train_ext)
        self.valid_tuple = BatchInput(
            data=valid_data, 
            sv_mask=valid_sv_mask, 
            ext_input=valid_ext)
        # create the datasets to batch the data, masks, and input
        self._train_ds = tf.data.Dataset.from_tensor_slices(self.train_tuple)
        self._valid_ds = tf.data.Dataset.from_tensor_slices(self.valid_tuple)
        # save the indices
        self.train_inds, self.valid_inds = train_inds, valid_inds

    def add_sample_validation(self, model_call):
        """Applies sample validation to a model-calling function.
        
        This decorator applies sample validation to a forward pass 
        through the model. It sets a certain proportion of the input 
        data elements to zero (heldout) and scales up the remaining 
        elements (heldin). It then computes NLL on the heldin 
        samples and the heldout samples separately. `nll_heldin` is 
        used for optimization and `nll_heldout` is used as a metric 
        to detect overfitting to spikes.

        Parameters
        ----------
        model_call : callable
            A callable function with inputs and outputs identical to 
            `LFADS.batch_call`.
        
        Returns
        -------
        callable
            A wrapper around `model_call` that computes heldin and 
            heldout NLL and returns the posterior parameters.
        """

        def sv_step(batch):
            # unpack the batch
            data, sv_mask, _ = batch
            heldin_mask, heldout_mask = sv_mask, tf.logical_not(sv_mask)
            # Only use CD where we are inferring rates (none inferred for IC segment)
            ic_enc_seq_len = self.cfg.MODEL.IC_ENC_SEQ_LEN
            unmaskable_data = data[:, :ic_enc_seq_len, :]
            maskable_data = data[:, ic_enc_seq_len:, :]
            # set the heldout data to zero and scale up heldin data
            wt_mask = tf.cast(heldin_mask, tf.float32) / self.sv_keep
            sv_masked_data = maskable_data * wt_mask
            heldin_data = tf.concat([unmaskable_data, sv_masked_data], axis=1)
            # perform the forward pass on the heldin data
            new_batch = BatchInput(
                data=heldin_data, 
                sv_mask=batch.sv_mask, 
                ext_input=batch.ext_input)
            logrates, posterior_params = model_call(new_batch)
            # compute the nll of the observed samples
            nll_heldin = self.neg_log_likelihood(maskable_data, logrates, wt_mask)
            if self.sv_keep < 1:
                # exclude the observed samples from the nll_heldout calculation
                wt_mask = tf.cast(heldout_mask, tf.float32) / (1-self.sv_keep)
                nll_heldout = self.neg_log_likelihood(maskable_data, logrates, wt_mask)
            else:
                nll_heldout = np.nan

            return nll_heldin, nll_heldout, posterior_params

        return sv_step


    def add_coordinated_dropout(self, model_call):
        """Applies coordinated dropout to a model-calling function.
        
        A decorator that applies coordinated dropout to a forward pass 
        through the model. It sets a certain proportion of the input 
        data elements to zero and scales up the remaining elements. When 
        the model is being trained, it can only backpropagate gradients 
        for matrix elements it didn't see at the input. The function 
        outputs a gradient mask that is incorporated by the sample 
        validation wrapper.

        Parameters
        ----------
        model_call : callable
            A callable function with inputs and outputs identical to 
            `LFADS.batch_call`.
        
        Returns
        -------
        callable
            A wrapper around `model_call` that blocks matrix elements 
            before the call and passes a mask to block gradients of the 
            observed matrix elements.

        """

        def block_gradients(input_data, keep_mask):
            keep_mask = tf.cast(keep_mask, tf.float32)
            block_mask = 1 - keep_mask
            return tf.stop_gradient(input_data * block_mask) + input_data * keep_mask

        def cd_step(batch):
            # Only use CD where we are inferring rates (none inferred for IC segment)
            ic_enc_seq_len = self.cfg.MODEL.IC_ENC_SEQ_LEN
            unmaskable_data = batch.data[:, :ic_enc_seq_len, :]
            maskable_data = batch.data[:, ic_enc_seq_len:, :]
            # samples a new coordinated dropout mask at every training step
            cd_mask = self.cd_input_dist.sample(sample_shape=tf.shape(maskable_data))
            pass_mask = self.cd_pass_dist.sample(sample_shape=tf.shape(maskable_data))
            grad_mask = tf.logical_or(tf.logical_not(cd_mask), pass_mask)
            # mask and scale post-CD input so it has the same sum as the original data
            cd_masked_data = maskable_data * tf.cast(cd_mask, tf.float32)
            cd_masked_data /= self.cd_keep
            # concatenate the data from the IC encoder segment if using
            cd_input = tf.concat([unmaskable_data, cd_masked_data], axis=1)
            # perform a forward pass on the cd masked data
            new_batch = BatchInput(
                data=cd_input, 
                sv_mask=batch.sv_mask, 
                ext_input=batch.ext_input)
            logrates, posterior_params = model_call(new_batch)
            # block the gradients with respect to the masked outputs
            logrates = block_gradients(logrates, grad_mask)
            return logrates, posterior_params

        return cd_step


    def batch_call(self, batch):
        """Performs the forward pass on a batch of input data.

        This is a wrapper around the forward pass of LFADS, meant to be 
        more compatible with the coordinated dropout and sample 
        validation wrappers.

        Parameters
        ----------
        batch : lfads_tf2.tuples.BatchInput
            A namedtuple contining tf.Tensors for spiking data, 
            external inputs, and a sample validation mask.

        Returns
        -------
        tf.Tensor
            A BxTxN tensor of log-rates, where B is the batch size, 
            T is the number of time steps, and N is the number of neurons.
        tuple of tf.Tensor
            Four tensors corresponding to the posteriors - `ic_mean`, 
            `ic_stddev`, `co_mean`, `co_stddev`.

        """
        input_data = LFADSInput(data=batch.data, ext_input=batch.ext_input)
        if self.cfg.TRAIN.EAGER_MODE: 
            output = self.call(input_data, use_logrates=True)
        else:
            output = self.graph_call(input_data, use_logrates=True)

        posterior_params = (
            output.ic_means, 
            output.ic_stddevs, 
            output.co_means, 
            output.co_stddevs)

        return output.rates, posterior_params


    def neg_log_likelihood(self, data, logrates, wt_mask=None):
        """Computes the log likelihood of the data, given 
        predicted rates. 

        This function computes the average negative log likelihood 
        of the spikes in this batch, given the rates that LFADS 
        predicts for the samples.

        Parameters
        ----------
        data : tf.Tensor
            A BxTxN tensor of spiking data.
        logrates : tf.Tensor
            A BxTxN tensor of log-rates.
        wt_mask : tf.Tensor
            A weighted mask to apply to the likelihoods.

        Returns
        -------
        tf.Tensor
            A scalar tensor representing the mean negative 
            log-likelihood of these spikes.        
        
        """
        if wt_mask is None:
            wt_mask = tf.ones_like(data)
        nll_all = tf.nn.log_poisson_loss(data, logrates, compute_full_loss=True)
        nll_masked = nll_all * wt_mask
        if self.cfg.TRAIN.NLL_MEAN:
            # Average over all elements of the data tensor
            nll = tf.reduce_mean(nll_masked)
        else:
            # Sum over inner dimensions, average over batch dimension
            nll = tf.reduce_mean(tf.reduce_sum(nll_masked, axis=[1,2]))
        return nll


    def weighted_kl_loss(self, ic_mean, ic_stddev, co_mean, co_stddev):
        """Computes the KL loss based on the priors.
        
        This function computes the weighted KL loss of all of 
        the posteriors. The KL of the initial conditions is computed 
        directly, but the KL of the controller output distributions
        is approximated via sampling.

        Parameters
        ----------
        ic_mean : tf.Tensor
            A BxIC_DIM tensor of initial condition means.
        ic_stddev : tf.Tensor
            A BxIC_DIM tensor of initial condition standard deviations.
        co_mean : tf.Tensor
            A BxTxCO_DIM tensor of controller output means.
        co_stddev : tf.Tensor
            A BxTxCO_DIM tensor of controller output standard deviations.
        
        Returns
        -------
        tf.Tensor
            A scalar tensor of the total KL loss of the model.

        """
        ic_post, co_post = self.make_posteriors(
            ic_mean, ic_stddev, co_mean, co_stddev)
        # Create the IC priors
        ic_prior_stddev = tf.exp(0.5 * self.ic_prior_logvar)
        ic_prior = tfd.MultivariateNormalDiag(self.ic_prior_mean, ic_prior_stddev)
        # compute KL for the IC's analytically
        ic_kl_batch = tfd.kl_divergence(ic_post, ic_prior)
        wt_ic_kl = tf.reduce_mean(ic_kl_batch) * self.kl_ic_weight

        if self.cfg.MODEL.CO_AUTOREG_PRIOR:
            # compute KL for the CO's via sampling
            wt_co_kl = 0.0
            if self.use_con:
                sample = co_post.sample()
                log_q = co_post.log_prob(sample)
                log_p = self.co_prior.log_prob(sample)
                wt_co_kl = tf.reduce_mean(log_q - log_p) * self.kl_co_weight
        else:
            # Create posterior without tfd.Independent (leave time in batch_shape)
            co_post = tfd.MultivariateNormalDiag(co_mean, co_stddev)
            # Create the CO priors
            co_prior_stddev = tf.exp(0.5 * self.co_prior_logvar)
            co_prior = tfd.MultivariateNormalDiag(
                self.co_prior_mean, co_prior_stddev)
            # Compute KL for CO's analytially, average across time and batch
            co_kl_batch = tfd.kl_divergence(co_post, co_prior)
            wt_co_kl = tf.reduce_mean(co_kl_batch) * self.kl_co_weight

        batch_size = ic_mean.shape[0]
        if self.training:
            self._update_metrics({
                'wt_ic_kl': wt_ic_kl, 
                'wt_co_kl': wt_co_kl}, batch_size)
        else:
            self._update_metrics({
                'val_wt_ic_kl': wt_ic_kl, 
                'val_wt_co_kl': wt_co_kl}, batch_size)

        return wt_ic_kl + wt_co_kl


    def _step(self, batch):
        """ Performs a step of training or validation.
        
        Depending on the state of the boolean `self.training` variable, 
        this function will either perform a step of training or a step
        of validation. This entails a forward pass through the model on 
        a batch of data, calculating and logging losses, and possibly 
        taking a training step.
        
        Parameters
        ----------
        batch : lfads_tf2.tuples.BatchInput
            A namedtuple contining tf.Tensors for spiking data, 
            external inputs, and a sample validation mask.
        
        """

        # ----- TRAINING STEP -----
        if self.training:
            with tf.GradientTape() as tape:
                # perform the forward pass, using SV and / or CD as necessary
                nll_heldin, nll_heldout, posterior_params = self.train_call(batch)

                rnn_losses = self.encoder.losses + self.decoder.losses
                l2 = tf.reduce_sum(rnn_losses) / \
                    (self.model_recurrent_size + tf.keras.backend.epsilon())
                kl = self.weighted_kl_loss(*posterior_params)

                loss = nll_heldin + self.l2_ramping_weight * l2 \
                    + self.kl_ramping_weight * kl

                # TODO: Move the lowd_readin to a subclass or treat this
                # the same way as the other L2 costs
                if self.cfg.MODEL.READIN_DIM > 0:
                    # Add the L2 cost of the readin, averaged over elements
                    kernel = self.lowd_readin.get_layer('lowd_readin').kernel
                    readin_size = tf.size(kernel, out_type=tf.float32)
                    loss += tf.reduce_sum(self.lowd_readin.losses) / readin_size
                
                scaled_loss = self.loss_scale * loss

            # compute gradients and descend
            gradients = tape.gradient(scaled_loss, self.trainable_variables)
            gradients, gnorm = tf.clip_by_global_norm(gradients, self.max_grad_norm)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            # record training statistics
            self._update_metrics({
                'loss': loss,
                'nll_heldin': nll_heldin,
                'nll_heldout': nll_heldout,
                'wt_kl': kl,
                'wt_l2': l2,
                'gnorm': gnorm,
            }, batch_size=batch.data.shape[0])

        # ----- VALIDATION STEP -----
        else:
            # perform the forward pass through the network
            nll_heldin, nll_heldout, posterior_params = self.val_call(batch)

            rnn_losses = self.encoder.losses + self.decoder.losses
            l2 = tf.reduce_sum(rnn_losses) / \
                (self.model_recurrent_size + tf.keras.backend.epsilon())
            kl = self.weighted_kl_loss(*posterior_params)

            loss = nll_heldin + self.l2_ramping_weight * l2 \
                + self.kl_ramping_weight * kl

            # TODO: Move the lowd_readin to a subclass or treat this
            # the same way as the other L2 costs
            if self.cfg.MODEL.READIN_DIM > 0:
                # Add the L2 cost of the readin, averaged over elements
                kernel = self.lowd_readin.get_layer('lowd_readin').kernel
                readin_size = tf.size(kernel, out_type=tf.float32)
                loss += tf.reduce_sum(self.lowd_readin.losses) / readin_size

            # record training statistics
            self._update_metrics({
                'val_loss': loss,
                'val_nll_heldin': nll_heldin,
                'val_nll_heldout': nll_heldout,
                'val_wt_kl': kl,
                'val_wt_l2': l2,
            }, batch_size=batch.data.shape[0])

    def make_posteriors(self, ic_mean, ic_stddev, co_mean, co_stddev):
        """ Creates posterior distributions from their parameters.
        
        Parameters
        ----------
        ic_mean : tf.Tensor
            A BxIC_DIM tensor of initial condition means.
        ic_stddev : tf.Tensor
            A BxIC_DIM tensor of initial condition standard deviations.
        co_mean : tf.Tensor
            A BxTxCO_DIM tensor of controller output means.
        co_stddev : tf.Tensor
            A BxTxCO_DIM tensor of controller output standard deviations.

        Returns
        -------
        tfd.MultivariateNormalDiag
            The initial condition posterior distribution.
        tfd.Independent(tfd.MultivariateNormalDiag)
            The controller output posterior distribution.

        """
        ic_post = tfd.MultivariateNormalDiag(ic_mean, ic_stddev)
        co_post = tfd.Independent(tfd.MultivariateNormalDiag(co_mean, co_stddev))
        return ic_post, co_post
    

    def _build_wrapped_call(self):
        """Assembles the forward pass using SV and CD wrappers.

        Conveniently wraps the forward pass of LFADS with coordinated
        dropout and sample validation to allow automatic application 
        of these paradigms.
        """
        train_call = self.batch_call
        if self.cd_keep < 1:
            train_call = self.add_coordinated_dropout(train_call)
        train_call = self.add_sample_validation(train_call)
        val_call = self.add_sample_validation(self.batch_call)

        if self.cfg.TRAIN.EAGER_MODE:
            self.train_call = train_call
            self.val_call = val_call
        else:
            mcfg = self.cfg.MODEL
            output_seq_len = mcfg.SEQ_LEN - mcfg.IC_ENC_SEQ_LEN
            data_shape = [None, mcfg.SEQ_LEN, mcfg.DATA_DIM]
            sv_mask_shape = [None, output_seq_len, mcfg.DATA_DIM]
            ext_input_shape = [None, output_seq_len, mcfg.EXT_INPUT_DIM]
            # single step of training or validation
            input_signature=[
                BatchInput(
                    tf.TensorSpec(shape=data_shape), 
                    tf.TensorSpec(shape=sv_mask_shape, dtype=tf.bool), 
                    tf.TensorSpec(shape=ext_input_shape))]
            self.train_call = tf.function(func=train_call, input_signature=input_signature)
            self.val_call = tf.function(func=val_call, input_signature=input_signature)


    def update_learning_rate(self):
        """Updates the learning rate of the optimizer. 
        
        Calculates the learning rate, based on model improvement
        over the last several epochs. After the last ramping epoch, we 
        start checking whether the current loss is worse than the worst
        in the previous few epochs. The exact number of epochs to 
        compare is determined by PATIENCE. If the current epoch is worse, 
        then we decline the learning rate by multiplying the decay factor. 
        This can only happen at most every PATIENCE epochs.

        Returns
        -------
        float
            The new learning rate.
        """

        cfg = self.cfg.TRAIN.LR
        prev_epoch = self.train_df.index[-1] if len(self.train_df) > 0 else 0
        if 0 < cfg.DECAY < 1 and prev_epoch > self.last_ramp_epoch + cfg.PATIENCE:
            # allow the learning rate to decay at a max of every cfg.PATIENCE epochs
            epochs_since_decay = (self.train_df.lr == self.train_df.at[prev_epoch, 'lr']).sum()
            if epochs_since_decay >= cfg.PATIENCE:
                # compare the current val_loss to the max over a window of previous epochs
                winmax_val_loss = self.train_df.iloc[-(cfg.PATIENCE+1):-1].val_loss.max()
                cur_val_loss = self.train_df.at[prev_epoch, 'val_loss']
                # if the current val_loss is greater than the max in the window, decay LR
                if cur_val_loss > winmax_val_loss:
                    new_lr = max([cfg.DECAY * self.learning_rate.numpy(), cfg.STOP])
                    self.learning_rate.assign(new_lr)
        # report the current learning rate to the metrics
        new_lr = self.learning_rate.numpy()
        self._update_metrics({'lr': new_lr})
        return new_lr


    def update_ramping_weights(self):
        """Updates the ramping weight variables. 
        
        In order to allow the model to learn more quickly, we introduce 
        regularization slowly by linearly increasing the ramping weight 
        as the model is training, to maximum ramping weights of 1. This 
        function computes the desired weight and assigns it to the ramping 
        variables, which are later used in the KL and L2 calculations.
        Ramping is determined by START_EPOCH, or the epoch to start 
        ramping, and INCREASE_EPOCH, or the number of epochs over which 
        to increase to full regularization strength. Ramping can occur 
        separately for KL and L2 losses, based on their respective 
        hyperparameters.

        """
        
        cfg = self.cfg.TRAIN
        cur_epoch = self.cur_epoch.numpy()
        def compute_weight(start_epoch, increase_epoch):
            if increase_epoch > 0:
                ratio = (cur_epoch - start_epoch) / (increase_epoch + 1)
                clipped_ratio = tf.clip_by_value(ratio, 0, 1)
                return tf.cast(clipped_ratio, tf.float32)
            else:
                return 1.0 if cur_epoch >= start_epoch else 0.0
        # compute the new weights
        kl_weight = compute_weight(cfg.KL.START_EPOCH, cfg.KL.INCREASE_EPOCH)
        l2_weight = compute_weight(cfg.L2.START_EPOCH, cfg.L2.INCREASE_EPOCH)
        # assign values to the ramping weight variables
        self.kl_ramping_weight.assign(kl_weight)
        self.l2_ramping_weight.assign(l2_weight)
        # report the current KL and L2 weights to the metrics
        self._update_metrics({'kl_wt': kl_weight, 'l2_wt': l2_weight})


    def train_epoch(self, loadable_data=None):
        """Trains LFADS for a single epoch.
        
        This function is designed to implement a single unit 
        of training, such that it may be called arbitrarily 
        by the user. It computes the desired loss weights, iterates 
        through the training dataset and validation dataset once, 
        computes and logs losses, checks stopping criteria, saves 
        checkpoints, and reports results in a dictionary. The 
        results of training this epoch are returned as a dictionary.

        Parameters
        ----------
        loadable_data : lfads_tf2.tuples.LoadableData
            A namedtuple containing the input data and external
            inputs for the training and validation sets. See 
            definition of LoadableData for more detail.

        Returns
        -------
        dict
            The results of all metrics evaluated during training.
 
        """

        if loadable_data is not None:
            self.load_datasets_from_arrays(loadable_data)

        cfg = self.cfg.TRAIN
        self.cur_epoch.assign_add(1)
        cur_epoch = self.cur_epoch.numpy()
        cur_lr = self.update_learning_rate()
        self.update_ramping_weights()
        epoch_header_template = ', '.join([
            'Current epoch: {cur_epoch}/{max_epochs}',
            'Steps completed: {cur_step}',
            'Patience: {patience}',
            'Status: {train_status}',
            'LR: {cur_lr:.2E}',
        ])
        epoch_header = epoch_header_template.format(**{
            'cur_epoch': cur_epoch, 
            'cur_step': self.cur_step.numpy(),
            'max_epochs': cfg.MAX_EPOCHS - 1, 
            'patience': self.cur_patience.numpy(), 
            'train_status': self.train_status, 
            'cur_lr': cur_lr,
        })
        self.lgr.info(epoch_header)
        # only use the remainder when it is the full batch
        if cfg.BATCH_SIZE > len(self.train_tuple.data):
            samples_per_epoch = len(self.train_tuple.data)
            drop_remainder = False
        else:
            samples_per_epoch = (len(self.train_tuple.data) // cfg.BATCH_SIZE) * cfg.BATCH_SIZE
            drop_remainder = True
        # only show progress bar when not in `TUNE_MODE`
        if not cfg.TUNE_MODE: 
            pbar = Progbar(samples_per_epoch, width=50, unit_name='sample')
        # main training loop over the batches
        for batch in self._train_ds.shuffle(10000).batch(cfg.BATCH_SIZE, drop_remainder=drop_remainder):
            self.training.assign(True)
            if cfg.EAGER_MODE:
                self._step(batch)
            else:
                self._graph_step(batch)
            self.cur_step.assign_add(1)
            if not cfg.TUNE_MODE:
                pbar.add(len(batch[0]))
        for val_batch in self._valid_ds.batch(cfg.BATCH_SIZE):
            self.training.assign(False)
            if cfg.EAGER_MODE:
                self._step(val_batch)
            else:
                self._graph_step(val_batch)
    
        def smth_metric(name, coef=0.7):
            # calculate exponentially smoothed value for a given metric
            cur_metric = self.all_metrics[name].result().numpy()
            prev_metric = self.prev_results.get('smth_' + name, cur_metric)
            smth_metric = (1 - coef) * prev_metric + coef * cur_metric
            return smth_metric

        # report smoothed values and epoch number to the metrics
        self._update_metrics({
            'smth_nll_heldin': smth_metric('nll_heldin'),
            'smth_nll_heldout': smth_metric('nll_heldout'),
            'smth_val_nll_heldin': smth_metric('val_nll_heldin'),
            'smth_val_nll_heldout': smth_metric('val_nll_heldout'),
        })

        # get the reset the metrics after each epoch (record before breaking the loop)
        results = {name: m.result().numpy() for name, m in self.all_metrics.items()}
        _ = [m.reset_states() for name, m in self.all_metrics.items()]
        results.update({'epoch': cur_epoch, 'step': self.cur_step.numpy()})

        # print the status of the metrics
        train_template = ' - '.join([
            "    loss: {loss:.3f}",
            "    nll_heldin: {nll_heldin:.3f}",
            "    nll_heldout: {nll_heldout:.3f}",
            "    wt_kl: {wt_kl:.2E}",
            "    wt_l2: {wt_l2:.2E}",
            "gnorm: {gnorm:.2E}",
        ])
        val_template = ' - '.join([
            "val_loss: {val_loss:.3f}",
            "val_nll_heldin: {val_nll_heldin:.3f}",
            "val_nll_heldout: {val_nll_heldout:.3f}",
            "val_wt_kl: {val_wt_kl:.2E}",
            "val_wt_l2: {val_wt_l2:.2E}",
        ])
        self.lgr.info(train_template.format(**results))
        self.lgr.info(val_template.format(**results))

        if self.cfg.TRAIN.LOG_HPS:
            # report the HPs for this stretch of training
            cfg_dict = flatten(yaml.safe_load(self.cfg.dump()))
            results.update(cfg_dict)

        # write the metrics and HPs to the in-memory `train_df` for evaluation
        new_results_df = pd.DataFrame({key: [val] for key, val in results.items()}).set_index('epoch')
        self.train_df = pd.concat([self.train_df, copy.deepcopy(new_results_df)])

        # ---------- Check all criteria that could stop the training loop ----------
        def check_max_epochs(train_df):
            pass_check = True
            if cur_epoch >= self.cfg.TRAIN.MAX_EPOCHS:
                pass_check = False
            return pass_check

        def check_nans(train_df):
            """ Check if training should stop because of NaN's """
            loss = train_df.at[cur_epoch, 'loss']
            val_loss = train_df.at[cur_epoch, 'val_loss']
            pass_check = True
            nan_found = np.isnan(loss) or np.isnan(val_loss)
            # Only stop for NaN loss when not in PBT_MODE
            if not cfg.PBT_MODE and nan_found:
                self.lgr.info("STOPPING: NaN found in loss.")
                pass_check = False
            return pass_check

        def check_lr(train_df):
            """ Check if training should stop because of the learning rate. """
            cur_lr = train_df.at[cur_epoch, 'lr']
            pass_check = True
            if cur_lr <= cfg.LR.STOP:
                self.lgr.info(f"STOPPING: Learning rate has reached {cfg.LR.STOP}.")
                pass_check = False
            return pass_check
        
        def check_earlystop(train_df):
            """ Check if training should stop because the validation loss has not improved. """
            best_epoch = -1
            pass_check = True
            if cur_epoch > self.last_ramp_epoch:
                # find the best epoch only after ramping is complete
                postramp_df = train_df[(train_df['kl_wt'] == 1) & (train_df['l2_wt'] == 1)]
                best_epoch = postramp_df.smth_val_nll_heldin.idxmin()
                # save a checkpoint if this model is the best and beyond `self.last_ramp_epoch`
                # use the `self.train_status` to report the status of early stopping
                if best_epoch == cur_epoch:
                    self.train_status = "<TRAINING>"
                    # save a checkpoint if this model is the best
                    self.lve_manager.save()
                else:
                    self.train_status = "<WAITING>"
            # stop training if `smth_val_nll` does not improve  after `cfg.PATIENCE` epochs
            bounds =  [best_epoch, self.last_ramp_epoch]
            self.cur_patience.assign(max([cur_epoch - max(bounds), 0]))
            if self.cur_patience.numpy() >= cfg.PATIENCE:
                self.lgr.info(f"STOPPING: No improvement in `smth_val_nll_heldin` for {cfg.PATIENCE} epochs.")
                pass_check = False
            return pass_check

        # Run all of the checks for stopping criterion
        check_funcs = [check_max_epochs, check_nans, check_lr, check_earlystop]
        checks_passed = [check_func(self.train_df) for check_func in check_funcs]
        # end the training loop if not all checks passed
        if not all(checks_passed):
            # Indicate that training is over
            results['done'] = True
            # Remove the last (ending) epoch from the `train_df`, in case we restart training
            self.train_df = self.train_df[:-1]
        else:
            # If all checks pass, save a checkpoint
            self.mrc_manager.save()
            # Save the results from the previous successful epoch for comparison
            self.prev_results = copy.deepcopy(results)
            # Add quotation marks so commas inside strings are ignored by csv parser
            convert_for_csv = lambda data: f'\"{data}\"' if type(data) == str else str(data)
            csv_output = [convert_for_csv(results[log_metric]) \
                for log_metric in self.logging_metrics + self.logging_hps]
            # Save the results of successful epochs to the CSV
            self.csv_lgr.info(','.join(csv_output))
        
        # log the metrics for tensorboard
        if self.cfg.TRAIN.USE_TB:

            def pca(data):
                # reduces the data to its first principal component for visualization
                pca_data = tf.reshape(data, (data.shape[0], -1)) if tf.rank(data) > 2 else data
                pca_obj = PCA(n_components=1)
                return pca_obj.fit(pca_data).transform(pca_data)

            def make_figure(data, rates, co_post):
                figure = plt.figure(figsize=(10,10))
                if co_post.event_shape[-1] > 0:
                    plt.subplot(2, 2, 1, title='Single Neuron')
                    plt.plot(data[0,:,0], label='spikes')
                    plt.plot(rates[0,:,0], label='rates')
                    plt.subplot(2, 2, 2, title='Sample Controller Output')
                    plt.plot(co_post.sample()[0,:,0])
                else:
                    plt.subplot(2, 1, 1, title='Single Neuron')
                    plt.plot(data[0,:,0], label='spikes')
                    plt.plot(rates[0,:,0], label='rates')
                plt.subplot(2, 2, 3, title='All Spikes')
                plt.imshow(tf.transpose(data[0,:,:]))
                plt.subplot(2, 2, 4, title='All Rates')
                plt.imshow(tf.transpose(rates[0,:,:]))
                return figure

            def figure_to_tf(figure):
                # save the plot to a PNG in memory
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close(figure)
                # convert the png buffer to TF image
                buf.seek(0)
                image = tf.image.decode_png(buf.getvalue(), channels=4)
                return tf.expand_dims(image, 0)

            with self.summary_writer.as_default():
                tf.summary.experimental.set_step(cur_epoch)
                with tf.name_scope('lfads'):
                    for name, value in results.items():
                        # record all of the results in tensorboard
                        if type(value) == float:
                            tf.summary.scalar(name, value)
                    if cur_epoch % 5 == 0:
                        # prepare the data for the forward pass - NOTE: no sample validation is used here.
                        train_input = LFADSInput(data=batch.data, ext_input=batch.ext_input)
                        val_input = LFADSInput(data=val_batch.data, ext_input=val_batch.ext_input)
                        # do a forward pass on all data in eager mode so we can log distributions to tensorboard
                        t_output = self.graph_call(train_input)
                        t_rates = t_output.rates
                        posterior_params = (
                            t_output.ic_means, 
                            t_output.ic_stddevs, 
                            t_output.co_means, 
                            t_output.co_stddevs)
                        t_ic_post, t_co_post = self.make_posteriors(*posterior_params)
                        v_output = self.graph_call(val_input)
                        v_rates = v_output.rates
                        posterior_params = (
                            v_output.ic_means, 
                            v_output.ic_stddevs, 
                            v_output.co_means, 
                            v_output.co_stddevs)
                        v_ic_post, v_co_post = self.make_posteriors(*posterior_params)
                        # make figures every fifth epoch - uses a different backend for matplotlib
                        plt.switch_backend('Agg')
                        train_fig = make_figure(self.train_tuple.data, t_rates, t_co_post)
                        tf.summary.image('train_ouput', figure_to_tf(train_fig))
                        val_fig = make_figure(self.valid_tuple.data, v_rates, v_co_post)
                        tf.summary.image('val_output', figure_to_tf(val_fig))
                        plt.switch_backend('agg')
                        # compute histograms of controller outputs
                        with tf.name_scope('prior'):
                            tf.summary.histogram('ic_mean', self.ic_prior_mean)
                            tf.summary.histogram('ic_stddev', tf.exp(0.5 * self.ic_prior_logvar))
                            if self.use_con:
                                if self.cfg.MODEL.CO_AUTOREG_PRIOR:
                                    tf.summary.histogram('priors/co_logtau', self.logtaus)
                                    tf.summary.histogram('priors/co_lognvar', self.lognvars)
                                else:
                                    tf.summary.histogram('priors/co_mean', self.co_prior_mean)
                                    tf.summary.histogram('priors/co_logvar', self.co_prior_logvar)
                        with tf.name_scope('post'):
                            with tf.name_scope('train'):
                                tf.summary.histogram('ic_mean', t_ic_post.mean())
                                tf.summary.histogram('ic_stddev', t_ic_post.stddev())
                                tf.summary.histogram('ic_mean_PC1', pca(t_ic_post.mean()))
                                tf.summary.histogram('ic_stddev_PC1', pca(t_ic_post.stddev()))
                                if self.use_con:
                                    tf.summary.histogram('co_mean', t_co_post.mean())
                                    tf.summary.histogram('co_stddev', t_co_post.stddev())
                                    tf.summary.histogram('co_mean_PC1', pca(t_co_post.mean()))
                                    tf.summary.histogram('co_stddev_PC1', pca(t_co_post.stddev()))
                            with tf.name_scope('valid'):
                                tf.summary.histogram('ic_mean', v_ic_post.mean())
                                tf.summary.histogram('ic_stddev', v_ic_post.stddev())
                                tf.summary.histogram('ic_mean_PC1', pca(v_ic_post.mean()))
                                tf.summary.histogram('ic_stddev_PC1', pca(v_ic_post.stddev()))
                                if self.use_con:
                                    tf.summary.histogram('co_mean', v_co_post.mean())
                                    tf.summary.histogram('co_stddev', v_co_post.stddev())
                                    tf.summary.histogram('co_mean_PC1', pca(v_co_post.mean()))
                                    tf.summary.histogram('co_stddev_PC1', pca(v_co_post.stddev()))

                self.summary_writer.flush()

        # remove NaN's, which can cause bugs for TensorBoard
        results = {key: val for key, val in results.items() if val != np.nan}
        return results


    def train(self, loadable_data=None):
        """Trains LFADS until any stopping criterion is reached. 
        
        Runs `LFADS.train_epoch` until it reports that training is 
        complete by including `results['done'] == True` in the results 
        dictionary.

        Parameters
        ----------
        loadable_data : lfads_tf2.tuples.LoadableData, optional
            A namedtuple containing the input data and external
            inputs for the training and validation sets. See 
            definition of LoadableData for more detail. By default,
            None uses data that has already been loaded.
        
        Returns
        -------
        dict
            The results of all metrics from the last epoch.
        """
        
        if loadable_data is not None:
            self.load_datasets_from_arrays(loadable_data)

        # train epochs until the `done` signal is recieved
        self.lgr.info("Train on {}, validate on {} samples".format(
            len(self.train_tuple.data), len(self.valid_tuple.data)))
        done = False
        while not done:
            results = self.train_epoch()
            done = results.get('done', False)

        # restore the best model if not using the most recent checkpoints
        if not self.cfg.TRAIN.PBT_MODE and self.lve_manager.latest_checkpoint is not None:
            self.restore_weights()

        # record that the model is trained
        self.is_trained = True

        return results

    def sample_and_average(self, 
                           loadable_data=None,
                           n_samples=50, 
                           batch_size=64,
                           ps_filename='posterior_samples.h5',
                           save=True,
                           merge_tv=False):
        """Saves rate estimates to the 'model_dir'.
        
        Performs a forward pass of LFADS, but passes multiple 
        samples from the posteriors, which can be used to get a 
        more accurate estimate of the rates. Saves all output 
        to posterior_samples.h5 in the `model_dir`.

        Parameters
        ----------
        loadable_data : lfads_tf2.tuples.LoadableData, optional
            A namedtuple containing the input data and external
            inputs for the training and validation sets. See 
            definition of LoadableData for more detail. By default,
            None uses data that has already been loaded.
        n_samples : int, optional
            The number of samples to take from the posterior 
            distribution for each datapoint, by default 50.
        batch_size : int, optional
            The number of samples per batch, by default 128.
        ps_filename : str, optional
            The name of the posterior sample file, by default
            'posterior_samples.h5'. Ignored if `save` is False.
        save : bool, optional
            Whether or not to save the posterior sampling output
            to a file, if False will return a tuple of 
            SamplingOutput. By default, True.
        merge_tv : bool, optional
            Whether to merge training and validation output, 
            by default False. Ignored if `save` is True.

        Returns
        -------
        SamplingOutput
            If save is True, return nothing. If save is False, 
            and merge_tv is false, retun SamplingOutput objects 
            training and validation data. If save is False and 
            merge_tv is True, return a single SamplingOutput 
            object.

        """

        if loadable_data is not None:
            self.load_datasets_from_arrays(loadable_data)

        output_file = os.path.join(
            self.cfg.TRAIN.MODEL_DIR, ps_filename)
        
        try:
            # remove any pre-existing posterior sampling file
            os.remove(output_file)
            self.lgr.info(
                f"Removing existing posterior sampling file at {output_file}")
        except OSError:
            pass

        if not self.is_trained:
            self.lgr.warn(
                "Performing posterior sampling on an untrained model.")
        
        # define merging and splitting utilities
        def merge_samp_and_batch(data, batch_dim):
            """ Combines the sample and batch dimensions """
            return tf.reshape(
                data, [n_samples * batch_dim] + tf.unstack(tf.shape(data)[2:]))

        def split_samp_and_batch(data, batch_dim):
            """ Splits up the sample and batch dimensions """
            return tf.reshape(
                data, [n_samples, batch_dim] + tf.unstack(tf.shape(data)[1:]))

        # ========== POSTERIOR SAMPLING ==========
        # perform sampling on both training and validation data
        for prefix, dataset in zip(['train_', 'valid_'], [self._train_ds, self._valid_ds]):
            data_len = len(self.train_tuple.data) if prefix == 'train_' else len(self.valid_tuple.data)

            # initialize lists to store rates
            all_outputs = []
            self.lgr.info("Posterior sample and average on {} segments.".format(data_len))
            if not self.cfg.TRAIN.TUNE_MODE:
                pbar = Progbar(data_len, width=50, unit_name='dataset')
            for batch in dataset.batch(batch_size):
                # unpack the batch
                data, _, ext_input = batch

                # pass data through low-dim readin for alignment compatibility
                if self.cfg.MODEL.READIN_DIM > 0 and not self.cfg.MODEL.ALIGN_MODE:
                    data = self.lowd_readin(data)

                # for each chop in the dataset, compute the initial conditions distribution
                ic_mean, ic_stddev, ci = self.encoder.graph_call(data)
                ic_post = tfd.MultivariateNormalDiag(ic_mean, ic_stddev)

                # sample from the posterior and merge sample and batch dimensions
                ic_post_samples = ic_post.sample(n_samples)
                ic_post_samples_merged = merge_samp_and_batch(ic_post_samples, len(data))

                # tile and merge the controller inputs and the external inputs
                ci_tiled = tf.tile(tf.expand_dims(ci, axis=0), [n_samples, 1, 1, 1])
                ci_merged = merge_samp_and_batch(ci_tiled, len(data))
                ext_tiled = tf.tile(tf.expand_dims(ext_input, axis=0), [n_samples, 1, 1, 1])
                ext_merged = merge_samp_and_batch(ext_tiled, len(data))

                # pass all samples into the decoder
                dec_input = DecoderInput(
                    ic_samp=ic_post_samples_merged,
                    ci=ci_merged,
                    ext_input=ext_merged)
                output_samples_merged = self.decoder.graph_call(dec_input)
                
                # average the outputs across samples
                output_samples = [split_samp_and_batch(t, len(data)) for t in output_samples_merged]
                output = [np.mean(t, axis=0) for t in output_samples]

                # aggregate for each batch
                non_averaged_outputs = [
                    ic_mean.numpy(),
                    tf.math.log(ic_stddev**2).numpy(),
                ]
                all_outputs.append(output + non_averaged_outputs)
                if not self.cfg.TRAIN.TUNE_MODE:
                    pbar.add(len(data))

            # collect the outputs for all batches and split them up into the appropriate variables
            all_outputs = list(zip(*all_outputs)) # transpose the list / tuple
            all_outputs = [np.concatenate(t, axis=0) for t in all_outputs]
            rates, co_means, co_stddevs, factors, gen_states, \
                gen_init, gen_inputs, con_states, \
                ic_post_mean, ic_post_logvar = all_outputs

            # return the output in an organized tuple
            samp_out = SamplingOutput(
                rates=rates, 
                factors=factors, 
                gen_states=gen_states, 
                gen_inputs=gen_inputs, 
                gen_init=gen_init, 
                ic_post_mean=ic_post_mean, 
                ic_post_logvar=ic_post_logvar, 
                ic_prior_mean=self.ic_prior_mean.numpy(), 
                ic_prior_logvar=self.ic_prior_logvar.numpy())

            # writes the output to the a file in the model directory
            with h5py.File(output_file, 'a') as hf:
                output_fields = list(samp_out._fields)
                for field in output_fields:
                        hf.create_dataset(
                            prefix+field, 
                            data=getattr(samp_out, field))
        # Save the indices if they exist
        if self.train_inds is not None and self.valid_inds is not None:
            with h5py.File(output_file, 'a') as hf:
                hf.create_dataset('train_inds', data=self.train_inds)
                hf.create_dataset('valid_inds', data=self.valid_inds)
        if not save:
            # If saving is disabled, load from the file and delete it
            output = load_posterior_averages(
                self.cfg.TRAIN.MODEL_DIR, merge_tv=merge_tv)
            os.remove(output_file)
            return output

    def restore_weights(self, lve=True):
        """
        Restores the weights of the model from the most 
        recent or least validation error checkpoint

        lve: bool (optional)
            whether to use the least validation error 
            checkpoint, by default True
        """
        # pass some data through the model to initialize weights
        mcfg = self.cfg.MODEL
        input_dim = mcfg.DATA_DIM if not mcfg.ALIGN_MODE \
            else mcfg.READIN_DIM
        data_shape = [10, mcfg.SEQ_LEN, input_dim]
        output_seq_len = mcfg.SEQ_LEN-mcfg.IC_ENC_SEQ_LEN
        ext_input_shape = [10, output_seq_len, mcfg.EXT_INPUT_DIM]
        noise = (
            np.ones(shape=data_shape, dtype=np.float32),
            np.ones(shape=ext_input_shape, dtype=np.float32),
        )
        self.call(noise)
        if lve:
            # restore the least validation error checkpoint
            self.lve_ckpt.restore(
                self.lve_manager.latest_checkpoint
            ).assert_nontrivial_match()
            self.lgr.info("Restoring the LVE model.")
        else:
            # restore the most recent checkpoint
            self.mrc_ckpt.restore(
                self.mrc_manager.latest_checkpoint
            ).assert_nontrivial_match()
            self.lgr.info("Restoring the most recent model.")
        self.is_trained=True

    def write_weights_to_h5(self, path):
        """Writes the weights of the model to an HDF5 file, 
        for directly transferring parameters from LF2 to lfadslite
        
        Parameters
        ----------
        path : str
            The path to the HDF5 file for saving the weights.
        """
        with h5py.File(path, 'w') as h5file:
            for variable in self.trainable_variables:
                array = variable.numpy()
                h5file.create_dataset(
                    variable.name,
                    array.shape,
                    data=array,
                )
