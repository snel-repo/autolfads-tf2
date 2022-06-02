from os import path

from yacs.config import CfgNode as CN

repo_path = path.dirname(path.dirname(path.dirname(path.realpath(__file__))))
DEFAULT_CONFIG_DIR = path.join(repo_path, "configs")


# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

_C.MODEL = CN()
# The dimension of the output distribution (i.e. number of neurons).
_C.MODEL.DATA_DIM = 50
# The number of neurons to hold out for co-smoothing
_C.MODEL.CS_DIM = 0
# The length of the input sequences.
_C.MODEL.SEQ_LEN = 100
# The number of timestemps to hold out for forward prediction
_C.MODEL.FP_LEN = 0
# The number of time steps to set aside for determining IC's
_C.MODEL.IC_ENC_SEQ_LEN = 0
# The number of external inputs to be fed into the LFADS model.
_C.MODEL.EXT_INPUT_DIM = 0
# The low-dim readin matrix for alignment compatibility (turn off with -1)
_C.MODEL.READIN_DIM = -1
# When true, disconnects the lowd_readin so it can be called externally
# before the aligner
_C.MODEL.ALIGN_MODE = False
# The hidden dimension of the bidirectional GRU layer that encodes
# the initial conditions for the generator network.
_C.MODEL.IC_ENC_DIM = 128
# The hidden dimension of the bidirectional GRU layer that encodes
# a portion of the inputs to the controller network.
_C.MODEL.CI_ENC_DIM = 128
# The hidden dimension of the controller GRUCell.
_C.MODEL.CON_DIM = 128
# The dimension of the controller output.
_C.MODEL.CO_DIM = 1
# The dimension of the initial condition distributions.
_C.MODEL.IC_DIM = 64
# The hidden dimension of the generator GRUCell.
_C.MODEL.GEN_DIM = 200
# The dimension of the learned factors.
_C.MODEL.FAC_DIM = 50
# Standard dropout throughout the model
_C.MODEL.DROPOUT_RATE = 0.05
# Rate of samples dropped at the input for CD
_C.MODEL.CD_RATE = 0.0
# Percentage of CD sample gradients that are allowed through the gradient block
_C.MODEL.CD_PASS_RATE = 0.0
# Percentage of spikes and rates held out for sample validation
_C.MODEL.SV_RATE = 0.0
# The seed for the sample validation masks
_C.MODEL.SV_SEED = 0
# When false, passes the means of the posterior distributions.
_C.MODEL.SAMPLE_POSTERIORS = True
# Whether to use an autoregressive prior. False uses Multivariate Gaussian
_C.MODEL.CO_AUTOREG_PRIOR = True
# The autoregressive constant on the controller output prior
_C.MODEL.CO_PRIOR_TAU = 10.0
# The noise variance on the controller output prior
_C.MODEL.CO_PRIOR_NVAR = 0.1
# The prior variance of IC's
_C.MODEL.IC_PRIOR_VAR = 0.1
# The minimum variance allowed for IC posteriors.
_C.MODEL.IC_POST_VAR_MIN = 0.0001
# Max absolute value of the GRU hidden states
_C.MODEL.CELL_CLIP = 5.0
# Number of bins to delay the controller input
_C.MODEL.CI_LAG = 1

_C.TRAIN = CN()

_C.TRAIN.DATA = CN()
# the directory to look for data
_C.TRAIN.DATA.DIR = "/snel/share/runs/PBT/paper-fix/lfadslite-testing/data_chaotic"
# all data files must have this prefix to be used
_C.TRAIN.DATA.PREFIX = "chaotic"

# number of epochs to wait before early stopping
_C.TRAIN.PATIENCE = 200
# number of samples per batch
_C.TRAIN.BATCH_SIZE = 128
# maximum number of training epochs
_C.TRAIN.MAX_EPOCHS = 10000
# the value above which gradients will be clipped
_C.TRAIN.MAX_GRAD_NORM = 200.0
# a multiplier for the loss
_C.TRAIN.LOSS_SCALE = 1.0e4
# whether to use mean or sum to calculate NLL for each sample
_C.TRAIN.NLL_MEAN = True

_C.TRAIN.LR = CN()
# the initial learning rate
_C.TRAIN.LR.INIT = 0.01
# the smallest allowable learning rate
_C.TRAIN.LR.STOP = 1e-5
# the learning rate decay factor
_C.TRAIN.LR.DECAY = 0.95
# the number of epochs to wait before decreasing LR
_C.TRAIN.LR.PATIENCE = 6
# the epsilon parameter for the optimzier
_C.TRAIN.ADAM_EPSILON = 1e-7

_C.TRAIN.L2 = CN()
# the epoch during which to start ramping L2 cost
_C.TRAIN.L2.START_EPOCH = 0
# the number of epochs to ramp L2 cost
_C.TRAIN.L2.INCREASE_EPOCH = 500
# the scale to weight the L2 of the ic encoder
_C.TRAIN.L2.IC_ENC_SCALE = 0.0
# the scale to weight the L2 of the ci encoder
_C.TRAIN.L2.CI_ENC_SCALE = 0.0
# the scale to weight the L2 of the generator
_C.TRAIN.L2.GEN_SCALE = 2000.0
# the scale to weight the L2 of the controller
_C.TRAIN.L2.CON_SCALE = 0.0
# the scale to weight the L2 of the linear readin
_C.TRAIN.L2.READIN_SCALE = 5e-3

_C.TRAIN.KL = CN()
# the epoch during which to start ramping KL cost
_C.TRAIN.KL.START_EPOCH = 0
# the number of epochs to ramp KL cost
_C.TRAIN.KL.INCREASE_EPOCH = 500
# the scale to weight the KL of the IC's
_C.TRAIN.KL.IC_WEIGHT = 1.0
# the scale to weight the KL of the con outputs
_C.TRAIN.KL.CO_WEIGHT = 1.0

# whether to train only the CI encoder and IC encoder
_C.TRAIN.ENCODERS_ONLY = False
# whether to use eager mode for the training step
_C.TRAIN.EAGER_MODE = False
# turns off console logging
_C.TRAIN.TUNE_MODE = False
# prevents restoring of the LVE model
_C.TRAIN.PBT_MODE = False
# saves the HPs to the training log - mainly used for PBT
_C.TRAIN.LOG_HPS = False

_C.TRAIN.USE_TB = True
_C.TRAIN.MODEL_DIR = "~/tmp/lfads_tf2/chaotic"
_C.TRAIN.OVERWRITE = False


def get_cfg_defaults():
    """Get default YACS config node for LFADS."""
    return _C.clone()
