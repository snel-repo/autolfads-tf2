from collections import namedtuple

"""
Classes that organize groups of tensors into namedtuples. 
Names and uses should be self-explanatory.
"""

LoadableData = namedtuple(
    'LoadableData', [
        'train_data', # training spiking data, TRxTxDATA_DIM
        'valid_data', # validation spiking data, VAxTxDATA_DIM
        'train_ext_input', # training external inputs, TRxTxEXT_INPUT_DIM, may be None
        'valid_ext_input', # validation external inputs, VAxTxEXT_INPUT_DIM, may be None
        'train_inds',
        'valid_inds',
    ])

BatchInput = namedtuple(
    'BatchInput', [
        'data', # spiking data, BxTxDATA_DIM
        'sv_mask', # sample validation max, BxTxDATA_DIM
        'ext_input', # external inputs, BxTxEXT_INPUT_DIM
    ])

EncoderOutput = namedtuple(
    'EncoderOutput', [
        'ic_mean', # means for the IC distributions, BxIC_DIM
        'ic_stddev', # stddev for the IC distributions, BxIC_DIM
        'ci', # controller inputs, BxTxCI_DIM
    ])

DecoderInput = namedtuple(
    'DecoderInput', [
        'ic_samp', # samples from IC distributions, BxIC_DIM
        'ci', # controller inputs, BxTxCI_DIM
        'ext_input', # external inputs, BxTxEXT_INPUT_DIM
    ])

DecoderRNNInput = namedtuple(
    'DecoderRNNInput', [
        'ci', # controller inputs, BxTxCI_DIM
        'ext_input', # external inputs, BxTxEXT_INPUT_DIM
    ])

DecoderState = namedtuple(
    'DecoderState', [
        'gen_state', # states of the generator RNN, BxGEN_DIM
        'con_state', # states of the controller RNN, BxCON_DIM
        'co_mean', # controller output means, BxCO_DIM
        'co_logvar', # controller output log-variances, BxCO_DIM
        'gen_input', # actual inputs to the generator, BxCO_DIM
        'factor', # latent factors produced by generator, BxFAC_DIM
    ])

DecoderOutput = namedtuple(
    'DecoderOutput', [
        'rates', # rate estimates, BxTxDATA_DIM
        'co_means', # controller output means, BxTxCO_DIM
        'co_stddevs', # controller output stddevs, BxTxCO_DIM
        'factors', # latent factors produced by generator, BxTxFAC_DIM
        'gen_states', # states of the generator RNN, BxTxGEN_DIM
        'gen_init', # initial states of the generator RNN, BxGEN_DIM
        'gen_inputs', # actual inputs to the generator, BxTxCO_DIM
        'con_states', # states of the controller RNN, BxTxCON_DIM
    ])

LFADSInput = namedtuple(
    'LFADSInput', [
        'data', # spiking data, BxTxDATA_DIM
        'ext_input', # external inputs, BxTxEXT_INPUT_DIM
    ])

LFADSOutput = namedtuple(
    'LFADSOutput', [
        'rates', # rate estimates, BxTxDATA_DIM
        'ic_means', # means for the IC distributions, BxIC_DIM
        'ic_stddevs', # stddev for the IC distributions, BxIC_DIM
        'co_means', # controller output means, BxTxCO_DIM
        'co_stddevs', # controller output stddevs, BxTxCO_DIM
        'factors', # latent factors produced by generator, BxTxFAC_DIM
        'gen_states', # states of the generator RNN, BxTxGEN_DIM
        'gen_init', # initial states of the generator RNN, BxGEN_DIM
        'gen_inputs', # actual inputs to the generator, BxTxCO_DIM
        'con_states', # states of the controller RNN, BxTxCON_DIM
    ])

SamplingOutput = namedtuple(
    'SamplingOutput', [
        'rates', # rate estimates, BxTxDATA_DIM
        'factors', # latent factors produced by generator, BxTxFAC_DIM
        'gen_states', # states of the generator RNN, BxTxGEN_DIM
        'gen_inputs', # actual inputs to the generator, BxTxCO_DIM
        'gen_init', # initial states of the generator RNN, BxGEN_DIM
        'ic_post_mean', # means for the IC posterior, BxIC_DIM
        'ic_post_logvar', # log-variance for the IC posterior, BxIC_DIM
        'ic_prior_mean', # means for the IC prior, BxIC_DIM
        'ic_prior_logvar', # log-variance for the IC prior, BxIC_DIM
    ])
