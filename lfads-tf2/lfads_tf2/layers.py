import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from lfads_tf2.initializers import make_variance_scaling, ones_zeros, variance_scaling
from lfads_tf2.regularizers import DynamicL2
from lfads_tf2.tuples import (
    DecoderInput,
    DecoderOutput,
    DecoderRNNInput,
    DecoderState,
    EncoderOutput,
)
from tensorflow.keras.layers import RNN, Bidirectional, Dense, Dropout, GRUCell, Layer

tfd = tfp.distributions
tfb = tfp.bijectors


class Encoder(Layer):
    """
    Defines the LFADS encoder, which takes as input a batch of time segments of
    binned neural spikes and converts them into a distribution over initial
    conditions for a generator network and a sequence of inputs to a controller
    network. Note that `Encoder` is a subclass of `tensorflow.keras.Model`.
    """

    def __init__(self, cfg_node):
        """Initializes an `Encoder` object. If no controller is used,
            the encoder will output only initial conditions.

        Parameters
        ----------
        cfg_node : yacs.config.CfgNode
            The configuration object for the model.
        """
        super(Encoder, self).__init__()
        # store the config
        self.cfg_node = cfg_node
        mcfg = cfg_node.MODEL
        # create the dropout layer
        self.dropout_rate = tf.Variable(mcfg.DROPOUT_RATE, trainable=False)
        self.dropout = Dropout(self.dropout_rate)
        self.use_con = all(
            [
                mcfg.CI_ENC_DIM > 0,
                mcfg.CON_DIM > 0,
                mcfg.CO_DIM > 0,
            ]
        )

        if self.use_con:
            # create the controller input BiGRU layer
            ci_enc_dim = mcfg.CI_ENC_DIM
            ci_enc_cell = ClippedGRUCell(
                ci_enc_dim,
                kernel_initializer=make_variance_scaling(
                    ci_enc_dim + (mcfg.DATA_DIM - mcfg.CS_DIM)
                ),
                recurrent_initializer=make_variance_scaling(
                    ci_enc_dim + (mcfg.DATA_DIM - mcfg.CS_DIM)
                ),
                bias_initializer=ones_zeros(ci_enc_dim),
                recurrent_regularizer=DynamicL2(cfg_node.TRAIN.L2.CI_ENC_SCALE),
                reset_after=False,
                name="ci_enc_gru_cell",
                clip_value=mcfg.CELL_CLIP,
            )
            self.ci_enc_bigru = Bidirectional(
                RNN(ci_enc_cell, return_sequences=True),
                merge_mode=None,
                name="ci_enc_bigru",
            )
        # create the initial condition BiGRU layer
        ic_enc_dim = mcfg.IC_ENC_DIM
        ic_enc_cell = ClippedGRUCell(
            ic_enc_dim,
            kernel_initializer=make_variance_scaling(
                ic_enc_dim + (mcfg.DATA_DIM - mcfg.CS_DIM)
            ),
            recurrent_initializer=make_variance_scaling(
                ic_enc_dim + (mcfg.DATA_DIM - mcfg.CS_DIM)
            ),
            bias_initializer=ones_zeros(ic_enc_dim),
            recurrent_regularizer=DynamicL2(cfg_node.TRAIN.L2.IC_ENC_SCALE),
            reset_after=False,
            name="ic_enc_gru_cell",
            clip_value=mcfg.CELL_CLIP,
        )
        self.ic_enc_bigru = Bidirectional(RNN(ic_enc_cell), name="ic_enc_bigru")
        # create the linear mappings to mean and logvar of initial conditions
        self.ic_mean_linear = Dense(
            mcfg.IC_DIM, kernel_initializer=variance_scaling, name="ic_mean_linear"
        )
        self.ic_logvar_linear = Dense(
            mcfg.IC_DIM, kernel_initializer=variance_scaling, name="ic_logvar_linear"
        )

        # ===== AUTOGRAPH FUNCTIONS =====
        if mcfg.READIN_DIM > 0:
            data_shape = [None, mcfg.SEQ_LEN, mcfg.READIN_DIM]
        else:
            data_shape = [None, mcfg.SEQ_LEN, mcfg.DATA_DIM]
        self.graph_call = tf.function(
            func=self.call,
            input_signature=[
                tf.TensorSpec(shape=data_shape),
                tf.TensorSpec(shape=[], dtype=tf.bool),
            ],
        )

    def build(self, input_shape):
        """Initializes the layer's variables.

        This method is overridden to handle creation of variables
        for the hidden states. It is called as soon as the model
        learns of the shape of its inputs. We use it here to
        initialize the initial state variables.

        Parameters
        ----------
        input_shape : tf.TensorShape
            The shape of the inputs to the encoder.
        """
        node = self.cfg_node

        # create the trainable initial states of the bigru layers
        self.ic_enc_h0 = tf.Variable(
            tf.zeros((1, 2, node.MODEL.IC_ENC_DIM)), name="ic_enc_bigru/h0"
        )
        if self.use_con:
            self.ci_enc_h0 = tf.Variable(
                tf.zeros((1, 2, node.MODEL.CI_ENC_DIM)), name="ci_enc_bigru/h0"
            )

        super(Encoder, self).build(input_shape)

    def call(self, data, training=False):
        """Performs the forward pass on the `Encoder` object.

        Parameters
        ----------
        data : np.array or tf.Tensor
            An B x T x N tensor of spike counts, where B is the
            batch dimension, T is the sequence length, and N is the
            number of neurons.
        training : bool, optional
            Whether to run the network in training mode, by default False

        Returns
        -------
        lfads_tf2.tuples.EncoderOutput
            A namedtuple containing the outputs of the Encoder,
            including the initial condition means and standard
            deviations, as well as the controller inputs.

        """

        mcfg = self.cfg_node.MODEL
        # check that the correct sequence length has been specified
        seq_len = mcfg.SEQ_LEN
        assert data.shape[1] == seq_len, (
            f"Sequence length specified in HPs ({seq_len}) "
            "must match data dim 1 ({data.shape[1]})."
        )

        # compute the generator IC's
        data = self.dropout(data, training)
        # option to use separate segment for IC encoding
        if mcfg.FP_LEN > 0:
            ic_enc_data = data[:, : -mcfg.FP_LEN, :]
            ci_enc_data = data[:, : -mcfg.FP_LEN, :]
        else:
            ic_enc_data = data
            ci_enc_data = data
        if mcfg.IC_ENC_SEQ_LEN > 0:
            ic_enc_data = ic_enc_data[:, : mcfg.IC_ENC_SEQ_LEN, :]
            ci_enc_data = ci_enc_data[:, mcfg.IC_ENC_SEQ_LEN :, :]
        if mcfg.CS_DIM > 0:
            ic_enc_data = ic_enc_data[:, :, : -mcfg.CS_DIM]
            ci_enc_data = ci_enc_data[:, :, : -mcfg.CS_DIM]
        # tile the initial states so they look like the data and unstack fw and bw
        ic_enc_h0 = tf.unstack(tf.tile(self.ic_enc_h0, [len(data), 1, 1]), axis=1)
        # choose subset of points for determining IC
        h_n = self.ic_enc_bigru(ic_enc_data, initial_state=ic_enc_h0)
        h_n_drop = self.dropout(h_n, training)
        ic_mean = self.ic_mean_linear(h_n_drop)
        ic_logvar = self.ic_logvar_linear(h_n_drop)
        ic_stddev = tf.sqrt(tf.exp(ic_logvar) + mcfg.IC_POST_VAR_MIN)

        if mcfg.CI_ENC_DIM > 0:
            # tile the initial states so they look like the data and unstack fw and bw
            ci_enc_h0 = tf.unstack(tf.tile(self.ci_enc_h0, [len(data), 1, 1]), axis=1)
            # compute encodings for the controller
            ci_fwd, ci_bwd = self.ci_enc_bigru(ci_enc_data, initial_state=ci_enc_h0)
            ci_len = seq_len - mcfg.FP_LEN - mcfg.IC_ENC_SEQ_LEN
            # add a lag to the controller input
            ci_fwd = tf.pad(ci_fwd, [[0, 0], [mcfg.CI_LAG, 0], [0, 0]])
            ci_bwd = tf.pad(ci_bwd, [[0, 0], [0, mcfg.CI_LAG], [0, 0]])
            # merge the forward and backward passes
            ci = tf.concat(
                [
                    tf.pad(ci_fwd[:, :ci_len, :], [[0, 0], [0, mcfg.FP_LEN], [0, 0]]),
                    tf.pad(ci_bwd[:, -ci_len:, :], [[0, 0], [0, mcfg.FP_LEN], [0, 0]]),
                ],
                axis=-1,
            )
        else:
            ci = tf.zeros_like(ci_enc_data)[:, :, : 2 * mcfg.CI_ENC_DIM]

        # return the output in an organized tuple
        output = EncoderOutput(
            ic_mean=ic_mean,
            ic_stddev=ic_stddev,
            ci=ci,
        )
        return output

    def get_config(self):
        """Get the configuration for the Encoder.

        See the TensorFlow documentation for an explanation of serialization:
        https://www.tensorflow.org/guide/keras/save_and_serialize#custom_objects

        Returns
        -------
        dict
            A dictionary containing the configuration node.
        """
        return {"cfg_node": self.cfg_node}

    @classmethod
    def from_config(cls, config):
        """Initialize an Encoder from this config.

        See the TensorFlow documentation for an explanation of serialization:
        https://www.tensorflow.org/guide/keras/save_and_serialize#custom_objects

        Returns
        -------
        lfads_tf2.layers.Encoder
            An Encoder from this config node.
        """
        return cls(**config)

    def update_config(self, config):
        """Updates the configuration of the Encoder.

        Updates configuration variables of the model.
        Primarily used for updating hyperparameters during PBT.

        Parameters
        ----------
        config : dict
            A dictionary containing the new configuration node.

        """

        self.cfg_node = node = config["cfg_node"]
        self.dropout_rate.assign(node.MODEL.DROPOUT_RATE)
        ic_enc_update = {
            "clip_value": node.MODEL.CELL_CLIP,
            "l2_recurrent_weight": node.TRAIN.L2.IC_ENC_SCALE,
        }
        self.ic_enc_bigru.forward_layer.cell.update_config(ic_enc_update)
        self.ic_enc_bigru.backward_layer.cell.update_config(ic_enc_update)
        if self.use_con:
            ci_enc_update = {
                "clip_value": node.MODEL.CELL_CLIP,
                "l2_recurrent_weight": node.TRAIN.L2.CI_ENC_SCALE,
            }
            self.ci_enc_bigru.forward_layer.cell.update_config(ci_enc_update)
            self.ci_enc_bigru.backward_layer.cell.update_config(ci_enc_update)


class Decoder(Layer):
    """
    Defines the LFADS decoder, which takes as input batches
    of samples from the initial condition posterior and,
    optionally, controller inputs. It evolves interacting
    controller and generator networks using these inputs and
    converts them to a sequence of controller output posterior
    distributions, generator states, factors, and Poisson rates.
    Note that `Decoder` is a subclass of `tensorflow.keras.Layer`.
    """

    def __init__(self, cfg_node):
        """Initializes a `Decoder` object.

        Parameters
        ----------
        cfg_node : yacs.config.CfgNode
            The configuration object for the model.
        """

        super(Decoder, self).__init__()

        self.cfg_node = cfg_node
        mcfg = cfg_node.MODEL

        # create the dropout layer
        self.dropout_rate = tf.Variable(mcfg.DROPOUT_RATE, trainable=False)
        self.dropout = Dropout(self.dropout_rate)

        self.use_con = all(
            [
                mcfg.CI_ENC_DIM > 0,
                mcfg.CON_DIM > 0,
                mcfg.CO_DIM > 0,
            ]
        )
        # create the linear mapping from ICs to gen_state
        self.ic_to_g0 = Dense(
            mcfg.GEN_DIM, kernel_initializer=variance_scaling, name="ic_to_g0"
        )
        # create the decoder RNN cell
        cell = DecoderCell(cfg_node)
        # create the decoding RNN
        self.rnn = RNN(cell, return_sequences=True, name="rnn")
        # create the mapping from factors to rates
        self.rate_linear = Dense(
            mcfg.DATA_DIM, kernel_initializer=variance_scaling, name="rate_linear"
        )

        # ===== AUTOGRAPH FUNCTIONS =====
        output_seq_len = mcfg.SEQ_LEN - mcfg.IC_ENC_SEQ_LEN
        ic_shape = [None, mcfg.IC_DIM]
        ci_shape = [None, output_seq_len, mcfg.CI_ENC_DIM * 2]
        ext_input_shape = [None, output_seq_len, mcfg.EXT_INPUT_DIM]
        self.graph_call = tf.function(
            func=self.call,
            input_signature=[
                DecoderInput(
                    tf.TensorSpec(shape=ic_shape),
                    tf.TensorSpec(shape=ci_shape),
                    tf.TensorSpec(shape=ext_input_shape),
                ),
                tf.TensorSpec(shape=[], dtype=tf.bool),
                tf.TensorSpec(shape=[], dtype=tf.bool),
            ],
        )

    def call(self, dec_input, training=False, use_logrates=False):
        """Performs the forward pass on the `Decoder` object.

        Parameters
        ----------
        dec_input : lfads_tf2.tuples.DecoderInput
            A namedtuple containing a batch of inputs to the decoder,
            Including samples from the IC distributions, controller
            inputs, and external inputs. See fields of DecoderInput
            for more detail.
        training :  bool, optional
            Whether to run the decoder in training mode.
        use_logrates : bool, optional
            Whether to return logrates, which are helpful for
            numerical stability of loss during training.

        Returns
        -------
        lfads_tf2.tuples.DecoderOutput
            All tensors output from the decoder. See fields of
            DecoderOutput for more detail.

        """

        # calculate initial generator state and pass it to the RNN with dropout rate
        gen_init = self.ic_to_g0(dec_input.ic_samp)
        gen_init_drop = self.dropout(gen_init, training)
        self.rnn.cell.setup_initial_state(gen_init, gen_init_drop)
        # perform dropout on the external inputs
        ext_input_drop = self.dropout(dec_input.ext_input, training)
        # prepare the decoder inputs and pass them to the rnn
        dec_rnn_input = DecoderRNNInput(ci=dec_input.ci, ext_input=ext_input_drop)
        states = self.rnn(dec_rnn_input, training=training)
        # separate the outputs of the decoder RNN
        gen_states, con_states, co_means, co_logvars, gen_inputs, factors = states
        co_stddevs = tf.exp(0.5 * co_logvars)
        # compute the rates
        logrates = self.rate_linear(factors)
        rates = tf.exp(logrates)

        # return the output in an organized tuple
        output = DecoderOutput(
            rates=logrates if use_logrates else rates,
            co_means=co_means,
            co_stddevs=co_stddevs,
            factors=factors,
            gen_states=gen_states,
            gen_init=gen_init,
            gen_inputs=gen_inputs,
            con_states=con_states,
        )
        return output

    def get_config(self):
        """Get the configuration for the Decoder.

        See the TensorFlow documentation for an explanation of serialization:
        https://www.tensorflow.org/guide/keras/save_and_serialize#custom_objects

        Returns
        -------
        dict
            A dictionary containing the configuration node.
        """

        return {"cfg_node": self.cfg_node}

    @classmethod
    def from_config(cls, config):
        """Initialize a Decoder from this config.

        See the TensorFlow documentation for an explanation of serialization:
        https://www.tensorflow.org/guide/keras/save_and_serialize#custom_objects

        Returns
        -------
        lfads_tf2.layers.Decoder
            A Decoder from this config node.
        """

        return cls(**config)

    def update_config(self, config):
        """Updates the configuration of the Decoder.

        Updates configuration variables of the model.
        Primarily used for updating hyperparameters during PBT.

        Parameters
        ----------
        config : dict
            A dictionary containing the new configuration node.

        """

        self.cfg_node = node = config["cfg_node"]
        self.dropout_rate.assign(node.MODEL.DROPOUT_RATE)
        self.rnn.cell.update_config(config)


class ClippedGRUCell(GRUCell):
    """Adds hidden state clipping, trainable initial states, and dynamic
    L2 penalty to the GRUCell implementation. Refer to tf.keras.GRUCell
    docs for args.
    """

    def __init__(self, *args, clip_value=np.inf, train_h0=False, **kwargs):
        """Creates a GRUCell with hidden state clipping and
        trainable hidden states.

        Parameters
        ----------
        clip_value : float, optional
            The value at which to clip the hidden states of the
            GRUCell, by default np.inf
        train_h0 : bool, optional
            Whether to use a trainable initial state, by default False
        """

        self.clip_value = tf.Variable(clip_value, trainable=False)
        self.train_h0 = train_h0
        # the GRUCell class applies the regularization
        super(ClippedGRUCell, self).__init__(*args, **kwargs)
        # TODO: find a better place to put this, where scope is perserved...
        # I wanted to put it in the build function, but there appear to be
        # cases in the bidirectional wrapper where get_initial_state is
        # called before build.
        if self.train_h0:
            with tf.name_scope(self.name_scope()):
                self.cell_h0 = self.add_weight(
                    shape=(1, self.units),
                    name="h0",
                    initializer="zeros",
                )

    def call(self, *args, **kwargs):
        """Performs one forward pass of the GRUCell.

        This function is a wrapper around tf.keras.layers.GRUCell.call
        that clips the value of the hidden state after every forward
        pass through the cell. Arguments are identical to the superclass.
        """
        h, _ = super(ClippedGRUCell, self).call(*args, **kwargs)
        h = tf.clip_by_value(h, -self.clip_value, self.clip_value)
        return h, [h]

    def get_initial_state(self, **kwargs):
        """Returns the initial state of the GRUCell.

        A wrapper around tf.keras.layers.GRUCell.get_initial_state
        that returns the trainable initial state if requested.
        """
        if self.train_h0:
            return tf.tile(self.cell_h0, [kwargs["batch_size"], 1])
        else:
            return super(ClippedGRUCell, self).get_initial_state(**kwargs)

    def get_config(self):
        """Get the configuration for the ClippedGRUCell.

        See the TensorFlow documentation for an explanation of serialization:
        https://www.tensorflow.org/guide/keras/save_and_serialize#custom_objects

        Returns
        -------
        dict
            A dictionary containing the configuration.
        """

        config = {
            "clip_value": self.clip_value.numpy(),
            "train_h0": self.train_h0,
        }
        base_config = super(GRUCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        """Initialize a ClippedGRUCell from this config.

        See the TensorFlow documentation for an explanation of serialization:
        https://www.tensorflow.org/guide/keras/save_and_serialize#custom_objects

        Returns
        -------
        lfads_tf2.layers.ClippedGRUCell
            A ClippedGRUCell from this config node.
        """

        from tensorflow.python.keras.layers import deserialize

        # re-create the custom regularizer object from its config
        recurrent_regularizer = deserialize(
            config.pop("recurrent_regularizer"), custom_objects={"DynamicL2": DynamicL2}
        )
        config["recurrent_regularizer"] = recurrent_regularizer
        # pass it with the rest of the config into the constructor
        return cls(**config)

    def update_config(self, config):
        """Updates the configuration of the ClippedGRUCell.

        Updates configuration variables of the layer.
        Primarily used for updating hyperparameters during PBT.

        Parameters
        ----------
        config : dict
            A dictionary containing the new configuration.

        """

        self.clip_value.assign(config["clip_value"])
        self.recurrent_regularizer.update_config(
            {
                "scale": config["l2_recurrent_weight"],
            }
        )


class KernelNormalizedDense(Dense):
    """Applies a row-normalized transformation."""

    def build(self, input_shape):
        """Creates the unnormalized kernel.

        A wrapper around tf.keras.layers.Dense.build that
        assigns the unnormalized kernel to a different variable.
        """
        super(KernelNormalizedDense, self).build(input_shape)
        self.unnormed_kernel = self.kernel

    def call(self, inputs):
        """Normalizes the linear kernel before the transformation

        A wrapper around tf.keras.layers.Dense.call that
        normalizes the kernel before applying the transfromation.
        """
        self.kernel = tf.nn.l2_normalize(self.unnormed_kernel, axis=0)
        outputs = super(KernelNormalizedDense, self).call(inputs)
        return outputs


class DecoderCell(Layer):
    """An RNN cell that incorporates interactions between
    the generator and controller networks.
    """

    def __init__(self, cfg_node, **kwargs):
        """Creates the DecoderCell object

        Parameters
        ----------
        cfg_node : yacs.config.CfgNode
            The configuration node for this the network.
        """

        self.cfg_node = cfg_node
        mcfg = cfg_node.MODEL

        # create the dropout layer
        self.dropout_rate = tf.Variable(mcfg.DROPOUT_RATE, trainable=False)
        self.dropout = Dropout(self.dropout_rate)

        # collect the appropriate dimensions from cfg_node
        self.ci_enc_dim = ci_enc_dim = mcfg.CI_ENC_DIM
        self.gen_units = gen_units = mcfg.GEN_DIM
        self.con_units = con_units = mcfg.CON_DIM
        self.fac_units = fac_units = mcfg.FAC_DIM
        self.co_units = co_units = mcfg.CO_DIM
        self.ext_units = ext_units = mcfg.EXT_INPUT_DIM
        self.clip_value = mcfg.CELL_CLIP
        self.con_l2_recurrent_weight = cfg_node.TRAIN.L2.CON_SCALE
        self.gen_l2_recurrent_weight = cfg_node.TRAIN.L2.GEN_SCALE
        self.gen_initial_state = None
        self.use_con = all(
            [
                ci_enc_dim > 0,
                con_units > 0,
                co_units > 0,
            ]
        )
        self.output_size = self.state_size = DecoderState(
            gen_state=gen_units,
            con_state=con_units,
            co_mean=co_units,
            co_logvar=co_units,
            gen_input=co_units + ext_units,
            factor=fac_units,
        )
        super(DecoderCell, self).__init__(**kwargs)

    def build(self, input_shapes):
        """Builds the layers of the cell.

        A wrapper around tf.keras.layers.Layer.build.

        Parameters
        ----------
        input_shapes : lfads_tf2.tuples.DecoderRNNInput
            The shapes of the inputs to the DecoderCell.
        """
        if self.use_con:
            # add the controller grucell
            self.con_gru_cell = ClippedGRUCell(
                self.con_units,
                kernel_initializer=variance_scaling,
                recurrent_initializer=variance_scaling,
                bias_initializer=ones_zeros(self.con_units),
                recurrent_regularizer=DynamicL2(self.con_l2_recurrent_weight),
                reset_after=False,
                name="con_gru_cell",
                clip_value=self.clip_value,
                train_h0=True,
            )
            # add the linear controller output mappings
            self.co_mean_linear = Dense(
                self.co_units,
                kernel_initializer=variance_scaling,
                name="co_mean_linear",
            )
            self.co_logvar_linear = Dense(
                self.co_units,
                kernel_initializer=variance_scaling,
                name="co_logvar_linear",
            )
        # add the generator
        self.gen_gru_cell = ClippedGRUCell(
            self.gen_units,
            kernel_initializer=variance_scaling,
            recurrent_initializer=variance_scaling,
            bias_initializer=ones_zeros(self.gen_units),
            recurrent_regularizer=DynamicL2(self.gen_l2_recurrent_weight),
            reset_after=False,
            name="gen_gru_cell",
            clip_value=self.clip_value,
            train_h0=False,
        )
        self.fac_linear = KernelNormalizedDense(
            self.fac_units,
            kernel_initializer=variance_scaling,
            use_bias=False,
            name="fac_linear",
        )
        super(DecoderCell, self).build(input_shapes)
        self.built = True

    def call(self, inputs, states, training=False):
        """Performs the forward pass on the DecoderCell.

        Parameters
        ----------
        inputs : lfads_tf2.tuples.DecoderRNNInput
            A namedtuple containing the inputs to the decoder
            RNN at a single time step, including controller
            inputs and external inputs. See DecoderRNNInput
            definition for more details.
        states : lfads_tf2.tuples.DecoderState
            A namedtuple containing the state of the decoder
            RNN at a single time step. See DecoderState
            definition for more details.
        training : bool, optional
            Whether to operate in training mode, by default False

        Returns
        -------
        lfads_tf2.tuples.DecoderState
            The state of the decoder cell after combining
            the current inputs with the previous state.
        """

        # unpack the inputs and states
        ci_step, ext_input_step = tf.nest.flatten(inputs)
        gen_state, con_state, co_mean, co_logvar, gen_input, factor = states

        if self.use_con:
            # compute controller inputs with dropout
            con_input = tf.concat([ci_step, factor], axis=-1)
            con_input_drop = self.dropout(con_input, training)
            # compute and store the next hidden state of the controller
            con_state, _ = self.con_gru_cell(con_input_drop, [con_state])
            # compute the distribution of the controller outputs at this timestep
            co_mean = self.co_mean_linear(con_state)
            co_logvar = self.co_logvar_linear(con_state)
            co_stddev = tf.exp(0.5 * co_logvar)
            # Generate controller outputs
            if self.cfg_node.MODEL.SAMPLE_POSTERIORS:
                # sample from the distribution of controller outputs
                co_dist = tfd.MultivariateNormalDiag(co_mean, co_stddev)
                con_output = co_dist.sample()
            else:
                # pass mean in deterministic mode
                con_output = co_mean
            # combine controller output with any external inputs
            gen_input = tf.concat([con_output, ext_input_step], axis=1)
        else:
            # if no controller is being used, can still provide ext inputs
            gen_input = ext_input_step
        # compute and store the next
        gen_state, _ = self.gen_gru_cell(gen_input, [gen_state])
        gen_state_drop = self.dropout(gen_state, training)
        factor = self.fac_linear(gen_state_drop)

        # pack the states up into a nested state
        new_states = DecoderState(
            gen_state, con_state, co_mean, co_logvar, gen_input, factor
        )
        return new_states, [new_states]

    def setup_initial_state(self, state, state_drop):
        """Passes the initial states into the generator and sets dropout.

        A convenience method that makes the generator state available
        to the `get_initial_state` function.

        Parameters
        ----------
        state : tf.Tensor
            The generator initial state.
        state_drop : tf.Tensor
            The generator initial state after dropout has been applied.
        """
        self.gen_initial_state = state
        self.gen_state_drop = state_drop

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Returns the initial state of the DecoderCell.

        Similar to `tf.keras.layers.GRUCell.get_initial_state`, it returns
        the initial state for a given shape of inputs.

        Returns
        -------
        lfads_tf2.tuples.DecoderState
            The initial state of the DecoderCell. See DecoderState
            definition for more details.
        """
        # Make sure the generator initial state has been computed
        assert isinstance(
            self.gen_initial_state, tf.Tensor
        ), "An initial state must be created for the generator."
        initial_factor = self.fac_linear(self.gen_state_drop)
        # Get the controller initial state
        if self.use_con:
            con_init = self.con_gru_cell.get_initial_state(
                inputs=inputs, batch_size=batch_size, dtype=dtype
            )
        else:
            con_init = tf.zeros([batch_size, self.con_units])
        # Return the nested state
        return DecoderState(
            gen_state=self.gen_initial_state,
            con_state=con_init,
            co_mean=tf.zeros([batch_size, self.co_units]),
            co_logvar=tf.zeros([batch_size, self.co_units]),
            gen_input=tf.zeros([batch_size, self.co_units + self.ext_units]),
            factor=initial_factor,
        )

    def get_config(self):
        """Get the configuration for the DecoderCell.

        See the TensorFlow documentation for an explanation of serialization:
        https://www.tensorflow.org/guide/keras/save_and_serialize#custom_objects

        Returns
        -------
        dict
            A dictionary containing the configuration.
        """

        config = {"cfg_node": self.cfg_node}
        base_config = super(DecoderCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        """Initialize a DecoderCell from this config.

        See the TensorFlow documentation for an explanation of serialization:
        https://www.tensorflow.org/guide/keras/save_and_serialize#custom_objects

        Returns
        -------
        lfads_tf2.layers.DecoderCell
            A DecoderCell from this config node.
        """

        return cls(**config)

    def update_config(self, config):
        """Updates the configuration of the DecoderCell.

        Updates configuration variables of the layer.
        Primarily used for updating hyperparameters during PBT.

        Parameters
        ----------
        config : dict
            A dictionary containing the new configuration.

        """

        self.cfg_node = node = config["cfg_node"]
        self.dropout_rate.assign(node.MODEL.DROPOUT_RATE)
        self.gen_gru_cell.update_config(
            {
                "clip_value": node.MODEL.CELL_CLIP,
                "l2_recurrent_weight": node.TRAIN.L2.GEN_SCALE,
            }
        )
        if self.use_con:
            self.con_gru_cell.update_config(
                {
                    "clip_value": node.MODEL.CELL_CLIP,
                    "l2_recurrent_weight": node.TRAIN.L2.CON_SCALE,
                }
            )


class AutoregressiveMultivariateNormal(tfd.Distribution):
    """Implements the prior distribution over controller outputs."""

    def __init__(self, logtaus, lognvars, co_dim, name=None):
        """Creates an AutoregressiveMultivariateNormal (ARMVN) distribution.

        Parameters
        ----------
        logtaus : tf.Variable
            The autocorrelation of the ARMVN distribution.
        lognvars : tf.Variable
            The noise variance of the ARMVN distribution.
        co_dim : int
            The dimension of the controller outputs.
        name : str, optional
            The name of the distibution, by default None
        """
        super(AutoregressiveMultivariateNormal, self).__init__(
            dtype=tf.float32,
            reparameterization_type=tfd.FULLY_REPARAMETERIZED,
            validate_args=True,
            allow_nan_stats=False,
            parameters={"logtaus": logtaus, "lognvars": lognvars},
            name=name,
        )
        self.logtaus = logtaus
        self.lognvars = lognvars

        step_spec = tf.TensorSpec(shape=[None, co_dim])
        self.step_log_prob = tf.function(
            func=self._step_log_prob, input_signature=[(step_spec, step_spec)]
        )

    def _step_log_prob(self, step):
        """Computes the log probability at a given timestep.

        Parameters
        ----------
        step : tuple of tf.Tensor
            A tuple of a sample and the previous sample to
            use for calculating the log probability.

        Returns
        -------
        tf.Tensor
            The log probability for this time step.
        """
        samp, prev_samp = tf.nest.flatten(step)
        # calculate alphas and process variances
        alphas = tf.math.exp(-1.0 / tf.math.exp(self.logtaus))
        logpvars = self.lognvars - tf.math.log(1 - alphas ** 2)
        # create a prior one step away from the observed sample
        first_sample = tf.reduce_all(tf.math.is_nan(prev_samp))
        if first_sample:
            means = 0.0
            stddevs = tf.math.exp(0.5 * logpvars)
        else:
            # calculate the new means and variances
            transform = tfb.Affine(scale_diag=alphas)
            means = transform.forward(prev_samp)
            stddevs = tf.math.exp(0.5 * self.lognvars)

        # return the log probability for each time step
        prior = tfd.MultivariateNormalDiag(means, stddevs)
        return prior.log_prob(samp)

    def _log_prob(self, sample):
        """Returns the log probability of a batch of CO samples.

        Parameters
        ----------
        sample : tf.Tensor
            A tensor of shape BxTxCO_DIM that represents samples
            of the controller output.

        Returns
        -------
        tf.Tensor
            A tensor of shape BATCH_SIZE indicating the
            log-likelihood of each controller output in the batch.
        """
        # create time-major pairs of samples and previous samples
        samples = tf.unstack(sample, axis=1)
        nan_sample = tf.fill(tf.shape(samples[0]), np.nan)
        prev_samples = [nan_sample] + samples[:-1]
        time_major_samples = tf.stack(samples, axis=0)
        time_major_prev_samples = tf.stack(prev_samples, axis=0)

        # sum the log-likelihood across the time dimension
        all_log_p = tf.map_fn(
            self.step_log_prob,
            (time_major_samples, time_major_prev_samples),
            # parallel_iterations=50
            dtype=tf.float32,
        )
        log_p = tf.reduce_sum(all_log_p, axis=0)

        return log_p

    def _event_shape(self):
        return self.logtaus.shape

    def _batch_shape(self):
        return tf.TensorShape([])
