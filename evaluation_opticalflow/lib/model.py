import functools
import operator
import os
from tqdm import tqdm
from plot_utils import plot_gt_preds
from traj_utils import relative_to_abs, vw_to_abs
from batches_data import get_batch
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras import models

class Model_Parameters(object):
    """Model parameters.
    """
    def __init__(self, add_attention=True, add_kp=False, add_social=False, output_representation='dxdy'):
        # -----------------
        # Observation/prediction lengths
        self.obs_len        = 8
        self.pred_len       = 12
        self.seq_len        = self.obs_len + self.pred_len
        self.add_kp         = add_kp
        self.add_social     = add_social
        self.add_attention  = add_attention
        self.stack_rnn_size = 2
        self.output_representation = output_representation
        self.output_var_dirs= 0
        # Key points
        self.kp_size        = 18
        # Optical flow
        self.flow_size      = 64
        # For training
        self.num_epochs     = 35
        self.batch_size     = 256  # batch size 512
        self.use_validation = True
        # Network architecture
        self.P              =   2 # Dimensions of the position vectors
        self.enc_hidden_size= 256                  # Default value in NextP
        self.dec_hidden_size= self.enc_hidden_size # Default value in NextP
        self.emb_size       = 128  # Default value in NextP
        self.dropout_rate   = 0.3 # Default value in NextP

        self.activation_func= tf.nn.tanh
        self.multi_decoder  = False
        self.modelname      = 'gphuctl'
        self.optimizer      = 'adam'
        self.initial_lr     = 0.01
        # MC dropout
        self.is_mc_dropout         = False
        self.mc_samples            = 20

""" Trajectory encoder through embedding+RNN.
"""
class TrajectoryEncoder(layers.Layer):
    def __init__(self, config):
        self.stack_rnn_size  = config.stack_rnn_size
        self.is_mc_dropout   = config.is_mc_dropout
        # xy encoder: [N,T1,h_dim]
        super(TrajectoryEncoder, self).__init__(name="trajectory_encoder")
        # Linear embedding of the observed trajectories (for each x,y)
        self.traj_xy_emb_enc = tf.keras.layers.Dense(config.emb_size,
            activation=config.activation_func,
            use_bias=True,
            name='position_embedding')
        # LSTM cell, including dropout
        # Stacked configuration.
        # Output is:
        # - sequence of h's (from the higher level): h1,h2,...
        # - last pair (h,c) for the first layer
        # - last pair (h,c) for the second layer
        # - ... and so on
        self.lstm_cells= [tf.keras.layers.LSTMCell(config.enc_hidden_size,
                name   = 'trajectory_encoder_cell',
                dropout= config.dropout_rate,
                recurrent_dropout=config.dropout_rate) for _ in range(self.stack_rnn_size)]
        self.lstm_cell = tf.keras.layers.StackedRNNCells(self.lstm_cells)

        # Recurrent neural network using the previous cell
        # Initial state is zero
        self.lstm      = tf.keras.layers.RNN(self.lstm_cell,
                return_sequences= True,
                return_state    = True)

    def call(self,traj_inputs,training=None):
        # Linear embedding of the observed trajectories
        x = self.traj_xy_emb_enc(traj_inputs)
        # Applies the position sequence through the LSTM
        # The training parameter is important for dropout
        return self.lstm(x,training=(training or self.is_mc_dropout))

""" Social encoding through embedding+RNN.
"""
class SocialEncoder(layers.Layer):
    def __init__(self, config):
        super(SocialEncoder, self).__init__(name="social_encoder")
        self.is_mc_dropout   = config.is_mc_dropout
        # Linear embedding of the social part
        self.traj_social_emb_enc = tf.keras.layers.Dense(config.emb_size,
            activation=config.activation_func,
            name='social_feature_embedding')
        # LSTM cell, including dropout
        self.lstm_cell = tf.keras.layers.LSTMCell(config.enc_hidden_size,
            name   = 'social_encoder_cell',
            dropout= config.dropout_rate,
            recurrent_dropout= config.dropout_rate)
        # Recurrent neural network using the previous cell
        self.lstm      = tf.keras.layers.RNN(self.lstm_cell,
            return_sequences= True,
            return_state    = True)

    def call(self,social_inputs,training=None):
        # Linear embedding of the observed trajectories
        x = self.traj_social_emb_enc(social_inputs)
        # Applies the position sequence through the LSTM
        return self.lstm(x,training=(training or self.is_mc_dropout))

""" Focal attention layer.
"""
class FocalAttention(layers.Layer):
    def __init__(self,config,M):
        super(FocalAttention, self).__init__(name="focal_attention")
        self.flatten  = tf.keras.layers.Flatten()
        self.reshape  = tf.keras.layers.Reshape((M, config.obs_len))

    def call(self,query, context):
        # query  : [N,D1]
        # context: [N,M,T,D2]
        # Get the tensor dimensions and check them
        _, D1       = query.get_shape().as_list()
        _, K, T, D2 = context.get_shape().as_list()
        assert D1 == D2
        # Expand [N,D1] -> [N,M,T,D1]
        query_aug = tf.tile(tf.expand_dims(tf.expand_dims(query, 1), 1), [1, K, T, 1])
        # Cosine similarity
        query_aug_norm = tf.nn.l2_normalize(query_aug, -1)
        context_norm   = tf.nn.l2_normalize(context,   -1)
        # Weights for pairs feature, time: [N, M, T]
        S         = tf.reduce_sum(tf.multiply(query_aug_norm, context_norm), 3)
        Wft       = self.reshape(tf.nn.softmax(self.flatten(S)))
        BQ        = tf.reduce_sum(tf.expand_dims(Wft, -1)*context,2)
        # Weigthts for features, maxed over time: [N,M]
        Sm        = tf.reduce_max(S, 2)
        Wf        = tf.nn.softmax(Sm)
        AQ        = tf.reduce_sum(tf.expand_dims(Wf, -1)*BQ,1)
        return tf.expand_dims(AQ,1), Wft

""" Custom model class for the encoding part (trajectory and context)
"""
class TrajectoryAndContextEncoder(tf.keras.Model):
    def __init__(self,config):
        super(TrajectoryAndContextEncoder, self).__init__(name="trajectory_context_encoder")
        self.add_social     = config.add_social
        self.add_attention  = config.add_attention
        self.stack_rnn_size = config.stack_rnn_size
        # Input layers
        obs_shape  = (config.obs_len,config.P)
        soc_shape  = (config.obs_len,config.flow_size)
        self.input_layer_traj = layers.Input(obs_shape,name="observed_trajectory")
        # Encoding: Positions
        self.traj_enc     = TrajectoryEncoder(config)
        if (self.add_attention and self.add_social):
            # In the case of handling social interactions, add a third input
            self.input_layer_social = layers.Input(soc_shape,name="social_features")
            # Encoding: Social interactions
            self.soc_enc            = SocialEncoder(config)
            # Get output layer now with `call` method
            self.out = self.call([self.input_layer_traj,self.input_layer_social])
        else:
            # Get output layer now with `call` method
            self.out = self.call([self.input_layer_traj])
        # Call init again. This is a workaround for being able to use summary
        super(TrajectoryAndContextEncoder, self).__init__(
            inputs=tf.cond(self.add_attention and self.add_social, lambda: [self.input_layer_traj,self.input_layer_social], lambda: [self.input_layer_traj]),
            outputs=self.out)

    def call(self,inputs,training=None):
        # inputs[0] is the observed part
        traj_obs_inputs  = inputs[0]
        if self.add_attention and self.add_social:
            # inputs[1] are the social interaction features
            soc_inputs     = inputs[1]
        # ----------------------------------------------------------
        # Encoding
        # ----------------------------------------------------------
        # Applies the position sequence through the LSTM: [N,T1,H]
        # In the case of stacked cells, output is:
        # sequence of outputs , last states (h,c) level 1, last states (h,c) level 2, ...
        outputs          = self.traj_enc(traj_obs_inputs,training=training)
        traj_h_seq       = outputs[0]
        traj_last_states = outputs[1:self.stack_rnn_size+1]
        # Get the hidden states and the last hidden state,+
        # separately, and add them to the lists
        enc_h_list          = [traj_h_seq]
        # ----------------------------------------------------------
        # Social interaccion (through optical flow)
        # ----------------------------------------------------------
        if self.add_social:
            # Applies the optical flow descriptor through the LSTM
            outputs = self.soc_enc(soc_inputs,training=training)
            soc_last_states = [outputs[1],outputs[2]]
            if self.add_attention:
                soc_h_seq       = outputs[0]
                # Get hidden states and the last hidden state, separately, and add them to the lists
                enc_h_list.append(soc_h_seq)
        # Pack all observed hidden states (lists) from all M features into a tensor
        # The final size should be [N,M,T_obs,h_dim]
        obs_enc_h          = tf.stack(enc_h_list, axis=1)
        if self.add_social:
            return traj_last_states,soc_last_states, obs_enc_h
        else:
            return traj_last_states, obs_enc_h

""" Custom LSTM cell class for our decoder
"""
class DecoderLSTMCell(tf.keras.layers.LSTMCell):
    def __init__(self, units, **kwargs):
        # Constructor of tf.contrib.rnn.BasicLSTMCell
        super(DecoderLSTMCell, self).__init__(units,**kwargs)
        # Forget bias (should be unit)
        self._forget_bias= 1.0

    # Overload the call function
    def call(self, inputs, states, training=None):
        # Get memory and carry state
        h_tm1 = states[0]
        c_tm1 = states[1]
        z  = tf.matmul(inputs, self.kernel)
        z += tf.matmul(h_tm1, self.recurrent_kernel)
        z  = tf.nn.bias_add(z, self.bias)

        # Split the z vector
        z0 = z[:, :self.units]
        z1 = z[:, self.units: 2 * self.units]
        z2 = z[:, 2 * self.units: 3 * self.units]
        z3 = z[:, 3 * self.units:]

        i = tf.sigmoid(z0)
        f = tf.sigmoid(z1)
        # New carry
        c = f * c_tm1 + i * self.activation(z2)
        o = tf.sigmoid(z3)
        # New state
        h = o * self.activation(c)
        return h, [h, c]

""" Trajectory decoder initializer.
    Allows
"""
class TrajectoryDecoderInitializer(tf.keras.Model):
    def __init__(self, config):
        super(TrajectoryDecoderInitializer, self).__init__(name="trajectory_decoder_initializer")
        self.is_mc_dropout  = config.is_mc_dropout
        self.add_social     = config.add_social
        # Dropout layer
        self.dropout        = tf.keras.layers.Dropout(config.dropout_rate)
        self.output_var_dirs= config.output_var_dirs
        # Linear embeddings from trajectory to hidden state
        self.traj_enc_h_to_dec_h = [tf.keras.layers.Dense(config.dec_hidden_size,
            activation=tf.keras.activations.relu,
            name='traj_enc_h_to_dec_h_%s'%i)  for i in range(self.output_var_dirs)]
        self.traj_enc_c_to_dec_c = [tf.keras.layers.Dense(config.dec_hidden_size,
            activation=tf.keras.activations.relu,
            name='traj_enc_c_to_dec_c_%s'%i)  for i in range(self.output_var_dirs)]
        if self.add_social:
            # Linear embeddings from social state to hidden state
            self.traj_soc_h_to_dec_h = [tf.keras.layers.Dense(config.dec_hidden_size,
                activation=tf.keras.activations.relu,
                name='traj_soc_h_to_dec_h_%s'%i)  for i in range(self.output_var_dirs)]
            self.traj_soc_c_to_dec_c = [tf.keras.layers.Dense(config.dec_hidden_size,
                activation=tf.keras.activations.relu,
                name='traj_soc_c_to_dec_c_%s'%i)  for i in range(self.output_var_dirs)]
        # Input layers
        input_shape      = (config.enc_hidden_size)
        self.input_h     = layers.Input(input_shape,name="trajectory_encoding_h")
        self.input_c     = layers.Input(input_shape,name="trajectory_encoding_c")
        if self.add_social:
            self.input_sh    = layers.Input(input_shape,name="social_encoding_h")
            self.input_sc    = layers.Input(input_shape,name="social_encoding_c")
            self.out         = self.call([[self.input_h,self.input_c],[self.input_sh,self.input_sc]])
            super(TrajectoryDecoderInitializer, self).__init__(
            inputs = [[self.input_h,self.input_c],[self.input_sh,self.input_sc]],
            outputs=self.out)
        else:
            self.out         = self.call([[self.input_h,self.input_c]])
            # Call init again. This is a workaround for being able to use summary
            super(TrajectoryDecoderInitializer, self).__init__(
                    inputs = [self.input_h,self.input_c],
                    outputs=self.out)

    # Call to the decoder
    def call(self, encoders_states, training=None):
        # Embeddings
        decoder_init_states = []
        traj_encoder_states  = encoders_states[0]
        if self.add_social:
            soc_encoder_states = encoders_states[1]
        # Append the single encoded states
        decoder_init_states.append(traj_encoder_states)
        for i in range(self.output_var_dirs):
            decoder_init_dh  = self.traj_enc_h_to_dec_h[i](traj_encoder_states[0])
            decoder_init_dc  = self.traj_enc_c_to_dec_c[i](traj_encoder_states[1])
            if self.add_social:
                decoder_init_dh = decoder_init_dh + self.traj_soc_h_to_dec_h[i](soc_encoder_states[0])
                decoder_init_dc = decoder_init_dc + self.traj_soc_c_to_dec_c[i](soc_encoder_states[1])
            decoder_init_h   = traj_encoder_states[0]+decoder_init_dh
            decoder_init_c   = traj_encoder_states[1]+decoder_init_dc
            decoder_init_states.append([decoder_init_h,decoder_init_c])
            decoder_init_h   = traj_encoder_states[0]-decoder_init_dh
            decoder_init_c   = traj_encoder_states[1]-decoder_init_dc
            decoder_init_states.append([decoder_init_h,decoder_init_c])
        return decoder_init_states

""" Trajectory decoder.
    Generates samples for the next position
"""
class TrajectoryDecoder(tf.keras.Model):
    def __init__(self, config):
        super(TrajectoryDecoder, self).__init__(name="trajectory_decoder")
        self.add_social     = config.add_social
        self.add_attention  = config.add_attention
        self.stack_rnn_size = config.stack_rnn_size
        self.is_mc_dropout  = config.is_mc_dropout
        # Linear embedding of the encoding resulting observed trajectories
        self.traj_xy_emb_dec = tf.keras.layers.Dense(config.emb_size,
            activation=config.activation_func,
            name='trajectory_position_embedding')
        # RNN cell
        self.dec_cell_traj  = tf.keras.layers.LSTMCell(config.dec_hidden_size,
            recurrent_initializer='glorot_uniform',
            name='trajectory_decoder_cell',
            dropout= config.dropout_rate,
            recurrent_dropout=config.dropout_rate
            )
        # RNN layer
        self.recurrentLayer = tf.keras.layers.RNN(self.dec_cell_traj,return_sequences=True,return_state=True)
        self.M = 1
        if (self.add_attention and self.add_social):
            self.M=self.M+1

        # Attention layer
        if (self.add_attention):
            self.focal_attention = FocalAttention(config,self.M)
        # Dropout layer
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate,name="dropout_decoder_h")
        # Mapping from h to positions
        self.h_to_xy = tf.keras.layers.Dense(config.P,
            activation=tf.identity,
            name='h_to_xy')
        # Input layers
        dec_input_shape      = (1,config.P)
        self.input_layer_pos = layers.Input(dec_input_shape,name="position")
        enc_last_state_shape = (config.dec_hidden_size)
        self.input_layer_hid1= layers.Input(enc_last_state_shape,name="initial_state_h")
        self.input_layer_hid2= layers.Input(enc_last_state_shape,name="initial_state_c")
        # Context shape: [N,M,T1,h_dim]
        ctxt_shape = (self.M,config.obs_len,config.enc_hidden_size)
        # Context input
        self.input_layer_ctxt = layers.Input(ctxt_shape,name="context")
        self.out = self.call(self.input_layer_pos,(self.input_layer_hid1,self.input_layer_hid2),self.input_layer_ctxt)
        # Call init again. This is a workaround for being able to use summary
        super(TrajectoryDecoder, self).__init__(
                    inputs= [self.input_layer_pos,self.input_layer_hid1,self.input_layer_hid2,self.input_layer_ctxt],
                    outputs=self.out)

    # Call to the decoder
    def call(self, dec_input, last_states, context, training=None):
        # Decoder inputs: position
        # Embedding
        decoder_inputs_emb = self.traj_xy_emb_dec(dec_input)
        # context: [N,1,h_dim]
        # query is the last h: [N,h_dim]
        query              = last_states[0]
        # Define the input here
        if self.add_attention:
            attention, Wft  = self.focal_attention(query, context)
            # Augmented input: [N,1,h_dim+emb]
            augmented_inputs= tf.concat([decoder_inputs_emb, attention], axis=2)
        else:
            Wft             = None
            # Input is just the embedded inputs
            augmented_inputs= decoder_inputs_emb
        # Application of the RNN: outputs are [N,1,dec_hidden_size],[N,dec_hidden_size],[N,dec_hidden_size]
        outputs    = self.recurrentLayer(augmented_inputs,initial_state=last_states,training=(training or self.is_mc_dropout))
        # Last h,c states
        cur_states = outputs[1:3]
        # Apply dropout layer on the h  state before mapping to positions x,y
        decoder_latent = self.dropout(cur_states[0],training=training)
        decoder_latent = tf.expand_dims(decoder_latent,1)
        # Mapping to positions x,y
        # decoder_out_xy = self.h_to_xy(decoder_latent)
        # Something new: we try to learn the residual to the constant velocity case
        decoder_out_xy = self.h_to_xy(decoder_latent) + dec_input
        return decoder_out_xy, cur_states, Wft

""" Trajectory classifier: during training, takes the observed trajectory and the prediction
    and predict
"""
class FullTrajectoryClassifier(tf.keras.Model):
    def __init__(self, config):
        super(FullTrajectoryClassifier, self).__init__(name="full_trajectory_classification")
        self.is_mc_dropout  = config.is_mc_dropout
        self.output_var_dirs= config.output_var_dirs
        self.output_samples = 2*config.output_var_dirs+1
        input_observed_shape= (config.enc_hidden_size)
        input_final_shape   = (config.P)
        # Inputs: hidden vector corresponding to the observations
        self.input_observed = keras.Input(shape=input_observed_shape, name="observed_trajectory_h")
        # Inputs: overall displacement along the second part of the trajectory
        self.input_final    = keras.Input(shape=input_final_shape, name="final_displacement")
        self.dense_layer_observed = tf.keras.layers.Dense(64, activation="relu", name="observed_dense")
        self.dense_layer_final    = tf.keras.layers.Dense(64, activation="relu", name="final_dense")
        self.classification_layer = layers.Dense(self.output_samples, activation="softmax", name="classication")
        # Get output layer now with `call` method
        self.out = self.call(self.input_observed,self.input_final)
        # Call init again. This is a workaround for being able to use summary
        super(FullTrajectoryClassifier, self).__init__(
                    inputs= [self.input_observed,self.input_final],
                    outputs=self.out)

    # Call to the classifier p(z|x,y)
    def call(self, observed_trajectory_h, final_position, training=None):
        # Linear embedding of the observed trajectories
        x = self.dense_layer_observed(observed_trajectory_h)
        y = self.dense_layer_final(final_position)
        interm = tf.concat([x,y], axis=1)
        return self.classification_layer(interm)

""" Observed trajectory classifier: during training, takes the observed trajectory and predict the class
"""
class ObservedTrajectoryClassifier(tf.keras.Model):
    def __init__(self, config):
        super(ObservedTrajectoryClassifier, self).__init__(name="observed_trajectory_classification")
        self.is_mc_dropout  = config.is_mc_dropout
        self.output_var_dirs= config.output_var_dirs
        self.output_samples = 2*config.output_var_dirs+1
        input_observed_shape= (config.enc_hidden_size)
        self.input_observed = keras.Input(shape=input_observed_shape, name="observed_trajectory_h")
        self.dense_layer_observed = tf.keras.layers.Dense(64, activation="relu", name="observed_dense")
        self.classification_layer = layers.Dense(self.output_samples, activation="softmax", name="classication")
        # Get output layer now with `call` method
        self.out = self.call(self.input_observed)
        # Call init again. This is a workaround for being able to use summary
        super(ObservedTrajectoryClassifier, self).__init__(
                    inputs= self.input_observed,
                    outputs=self.out)

    # Call to the classifier p(z|x,y)
    def call(self, observed_trajectory_h, training=None):
        # Linear embedding of the observed trajectories
        x = self.dense_layer_observed(observed_trajectory_h)
        return self.classification_layer(x)


# The main class
class TrajectoryEncoderDecoder():
    # Constructor
    def __init__(self,config):
        # Flags for considering social interations
        self.add_social     = config.add_social
        self.stack_rnn_size = config.stack_rnn_size
        self.output_samples = 2*config.output_var_dirs+1
        self.output_var_dirs= config.output_var_dirs

        #########################################################################################
        # The components of our model are instantiated here
        # Encoder: Positions and context
        self.enc = TrajectoryAndContextEncoder(config)
        self.enc.summary()
        # Classifier p(z|x)
        self.obs_classif = ObservedTrajectoryClassifier(config)
        self.obs_classif.summary()
        # Encoder to decoder initialization
        self.enctodec = TrajectoryDecoderInitializer(config)
        self.enctodec.summary()
        # Decoder
        self.dec = TrajectoryDecoder(config)
        self.dec.summary()
        #########################################################################################

        # Optimization scheduling
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                config.initial_lr,
                decay_steps=100000,
                decay_rate=0.96,
                staircase=True)

        # Instantiate an optimizer to train the models.
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
        self.optimizer = tf.keras.optimizers.Adadelta(learning_rate=lr_schedule)

        # Instantiate the loss operator
        #self.loss_fn = keras.losses.MeanSquaredError()
        self.loss_fn       = keras.losses.LogCosh()
        self.loss_fn_local = keras.losses.LogCosh(keras.losses.Reduction.NONE)

    # Trick to reset the weights: We save them and reload them
    def save_tmp(self):
        self.enc.save_weights('tmp_enc.h5')
        self.enctodec.save_weights('tmp_enctodec.h5')
        self.dec.save_weights('tmp_dec.h5')
        self.obs_classif.save_weights('tmp_obs_classif.h5')
    def load_tmp(self):
        self.enc.load_weights('tmp_enc.h5')
        self.enctodec.load_weights('tmp_enctodec.h5')
        self.dec.load_weights('tmp_dec.h5')
        self.obs_classif.load_weights('tmp_obs_classif.h5')

    # Single training/testing step, for one batch: batch_inputs are the observations, batch_targets are the targets
    def batch_step(self, batch_inputs, batch_targets, metrics, training=True):
        traj_obs_inputs = batch_inputs[0]
        # Last observed position from the trajectory
        traj_obs_last = traj_obs_inputs[:, -1]
        # Variables to be trained
        variables = self.enc.trainable_weights + self.enctodec.trainable_weights + self.dec.trainable_weights +self.obs_classif.trainable_weights
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        # The total loss will be accumulated on this variable
        loss_value = 0
        with tf.GradientTape() as g:
            #########################################################################################
            # Encoding is done here
            # Apply trajectory and context encoding
            if self.add_social:
                traj_last_states, soc_last_states, context = self.enc(batch_inputs, training=training)
                traj_cur_states_set = self.enctodec([traj_last_states[0],soc_last_states])
            else:
                traj_last_states, context = self.enc(batch_inputs, training=training)
                # Returns a set of self.output_samples possible initializing states for the decoder
                # Each value in the set is a pair (h,c) for the low level LSTM in the stack
                traj_cur_states_set = self.enctodec([traj_last_states[0]])
            # Apply the classifiers
            obs_classif_logits = self.obs_classif(traj_last_states[0][0])

            #########################################################################################
            # Decoding is done here
            # Iterate over these possible initializing states
            losses = []
            for k in range(self.output_samples):
                # Sample-wise loss values
                loss_values         = 0
                # Decoder state is initialized here
                traj_cur_states     = traj_cur_states_set[k]
                # The first input to the decoder is the last observed position [Nx1xK]
                dec_input = tf.expand_dims(traj_obs_last, 1)
                # Iterate over timesteps
                for t in range(0, batch_targets.shape[1]):
                    # ------------------------ xy decoder--------------------------------------
                    # passing enc_output to the decoder
                    t_pred, dec_states, __ = self.dec(dec_input,traj_cur_states,context,training=training)
                    t_target               = tf.expand_dims(batch_targets[:, t], 1)
                    # Loss
                    loss_values += (batch_targets.shape[1]-t)*self.loss_fn_local(t_target, t_pred)
                    if training==True:
                        # Using teacher forcing [Nx1xK]
                        # Teacher forcing - feeding the target as the next input
                        dec_input = tf.expand_dims(batch_targets[:, t], 1)
                    else:
                        # Next input is the last predicted position
                        dec_input = t_pred
                    # Update the states
                    traj_cur_states = dec_states
                # Keep loss values for all self.output_samples cases
                losses.append(tf.squeeze(loss_values,axis=1))
            # Stack into a tensor batch_size x self.output_samples
            losses          = tf.stack(losses, axis=1)
            closest_samples = tf.math.argmin(losses, axis=1)
            softmax_samples = tf.nn.softmax(-losses/0.01, axis=1)
            #########################################################################################

            #########################################################################################
            # Losses are accumulated here
            metrics['obs_classif_sca'].update_state(closest_samples,obs_classif_logits)
            loss_value  += 0.005* tf.reduce_sum(tf.keras.losses.kullback_leibler_divergence(softmax_samples,obs_classif_logits))/losses.shape[0]

            # Get the vector of losses at the minimal value for each sample of the batch
            losses_at_min= tf.gather_nd(losses,tf.stack([range(losses.shape[0]),closest_samples],axis=1))
            # Sum over the samples, divided by the batch size
            loss_value  += tf.reduce_sum(losses_at_min)/losses.shape[0]
            # TODO: tune this value in a more principled way?
            # L2 weight decay
            loss_value  += tf.add_n([ tf.nn.l2_loss(v) for v in variables
                        if 'bias' not in v.name ]) * 0.0008
            #########################################################################################


        if training==True:
            # Get the gradients
            grads = g.gradient(loss_value, variables)
            # Run one step of gradient descent
            self.optimizer.apply_gradients(zip(grads, variables))
        # Average loss over the predicted times
        batch_loss = (loss_value / int(batch_targets.shape[1]))
        return batch_loss

    # Single training/testing step, for one batch: training the classifier
    def batch_step_classifier(self, batch_inputs, batch_targets, metrics, training=True):
        traj_obs_inputs = batch_inputs[0]
        # Last observed position from the trajectory
        traj_obs_last = traj_obs_inputs[:, -1]
        # Variables to be trained
        variables = self.obs_classif.trainable_weights
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        # The total loss will be accumulated on this variable
        loss_value = 0
        with tf.GradientTape() as g:
            #########################################################################################
            # Encoding is done here
            # Apply trajectory and context encoding
            if self.add_social:
                traj_last_states, soc_last_states, context = self.enc(batch_inputs, training=False)
                traj_cur_states_set = self.enctodec([traj_last_states[0],soc_last_states])
            else:
                traj_last_states, context = self.enc(batch_inputs, training=False)
                # Returns a set of self.output_samples possible initializing states for the decoder
                # Each value in the set is a pair (h,c) for the low level LSTM in the stack
                traj_cur_states_set = self.enctodec([traj_last_states[0]])
            # Apply the classifiers
            obs_classif_logits = self.obs_classif(traj_last_states[0][0])
            #########################################################################################
            # Decoding is done here
            # Iterate over these possible initializing states
            losses = []
            for k in range(self.output_samples):
                # Sample-wise loss values
                loss_values         = 0
                # Decoder state is initialized here
                traj_cur_states     = traj_cur_states_set[k]
                # The first input to the decoder is the last observed position [Nx1xK]
                dec_input = tf.expand_dims(traj_obs_last, 1)
                # Iterate over timesteps
                for t in range(0, batch_targets.shape[1]):
                    # ------------------------ xy decoder--------------------------------------
                    # passing enc_output to the decoder
                    t_pred, dec_states, __ = self.dec(dec_input,traj_cur_states,context,training=training)
                    t_target               = tf.expand_dims(batch_targets[:, t], 1)
                    # Loss
                    loss_values += (batch_targets.shape[1]-t)*self.loss_fn_local(t_target, t_pred)
                    # Next input is the last predicted position
                    dec_input = t_pred
                    # Update the states
                    traj_cur_states = dec_states
                # Keep loss values for all self.output_samples cases
                losses.append(tf.squeeze(loss_values,axis=1))
            # Stack into a tensor batch_size x self.output_samples
            losses          = tf.stack(losses, axis=1)
            closest_samples = tf.math.argmin(losses, axis=1)
            softmax_samples = tf.nn.softmax(-losses/0.01, axis=1)
            #########################################################################################

            #########################################################################################
            # Losses are accumulated here
            metrics['obs_classif_sca'].update_state(closest_samples,obs_classif_logits)
            loss_value  += 0.005* tf.reduce_sum(tf.keras.losses.kullback_leibler_divergence(softmax_samples,obs_classif_logits))/losses.shape[0]
            # TODO: tune this value in a more principled way?
            # L2 weight decay
            loss_value  += tf.add_n([ tf.nn.l2_loss(v) for v in variables
                        if 'bias' not in v.name ]) * 0.0008
            #########################################################################################


        if training==True:
            # Get the gradients
            grads = g.gradient(loss_value, variables)
            # Run one step of gradient descent
            self.optimizer.apply_gradients(zip(grads, variables))
        # Average loss over the predicted times
        batch_loss = (loss_value / int(batch_targets.shape[1]))
        return batch_loss

    # Prediction (testing) for one batch
    def batch_predict(self, batch_inputs, n_steps, mc_samples=1):
        traj_obs_inputs = batch_inputs[0]
        # Last observed position from the trajectories
        traj_obs_last     = traj_obs_inputs[:, -1]
        all_samples       = []
        all_probabilities = []
        for i in range(mc_samples):
            # Feed-forward start here
            if self.add_social:
                traj_last_states, soc_last_states, context = self.enc(batch_inputs, training=False)
                traj_cur_states_set = self.enctodec([traj_last_states[0],soc_last_states])
            else:
                traj_last_states, context = self.enc(batch_inputs, training=False)
                # Returns a set of self.output_samples possible initializing states for the decoder
                # Each value in the set is a pair (h,c) for the low level LSTM in the stack
                traj_cur_states_set = self.enctodec([traj_last_states[0]])
            # Apply the classifier to the encoding of the observed part
            obs_classif_logits = self.obs_classif(traj_last_states[0][0])

            # This will store the trajectories and the attention weights
            traj_pred_set       = []
            att_weights_pred_set= []

            # Iterate over these possible initializing states
            for k in range(self.output_samples):
                # List for the predictions and attention weights
                traj_pred       = []
                att_weights_pred= []
                # Decoder state is initialized here
                traj_cur_states     = traj_cur_states_set[k]
                # The first input to the decoder is the last observed position [Nx1xK]
                dec_input = tf.expand_dims(traj_obs_last, 1)
                # Iterate over timesteps
                for t in range(0, n_steps):
                    # ------------------------ xy decoder--------------------------------------
                    # Passing enc_output to the decoder
                    t_pred, dec_states, wft = self.dec(dec_input,traj_cur_states,context,training=False)
                    # Next input is the last predicted position
                    dec_input = t_pred
                    # Add it to the list of predictions
                    traj_pred.append(t_pred)
                    att_weights_pred.append(wft)
                    # Reuse the hidden states for the next step
                    traj_cur_states = dec_states
                traj_pred        = tf.squeeze(tf.stack(traj_pred, axis=1))
                att_weights_pred = tf.squeeze(tf.stack(att_weights_pred, axis=1))
                traj_pred_set.append(traj_pred)
                att_weights_pred_set.append(att_weights_pred)
            all_samples.append([traj_pred_set,att_weights_pred_set])
            all_probabilities.append(obs_classif_logits)
        return all_samples, all_probabilities

    # Training loop
    def training_loop(self,train_data,val_data,config,checkpoint,checkpoint_prefix):
        num_batches_per_epoch= train_data.get_num_batches()
        train_loss_results   = []
        val_loss_results     = []
        val_metrics_results  = {'ade': [], 'fde': [], 'obs_classif_accuracy': []}
        train_metrics_results= {'obs_classif_accuracy': []}
        best                 = {'ade':999999, 'fde':0, 'batchId':-1}
        train_metrics        = {'obs_classif_sca':keras.metrics.SparseCategoricalAccuracy()}
        val_metrics          = {'obs_classif_sca':keras.metrics.SparseCategoricalAccuracy()}

        # Training the main system
        for epoch in range(config.num_epochs):
            print('Epoch {}.'.format(epoch + 1))
            # Cycle over batches
            total_loss = 0
            num_batches_per_epoch = train_data.get_num_batches()
            for idx,batch in tqdm(train_data.get_batches(config.batch_size, num_steps = num_batches_per_epoch, shuffle=True), total = num_batches_per_epoch, ascii = True):
                # Format the data
                batch_inputs, batch_targets = get_batch(batch, config)
                # Run the forward pass of the layer.
                # Compute the loss value for this minibatch.
                batch_loss = self.batch_step(batch_inputs, batch_targets, train_metrics, training=True)
                total_loss+= batch_loss
            # End epoch
            total_loss = total_loss / num_batches_per_epoch
            train_loss_results.append(total_loss)

            # Saving (checkpoint) the model every 2 epochs
            if (epoch + 1) % 2 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)

            # Display information about the current state of the training loop
            print('[TRN] Epoch {}. Training loss {:.4f}'.format(epoch + 1, total_loss ))
            print('[TRN] Training accuracy of classifier p(z|x)   {:.4f}'.format(float(train_metrics['obs_classif_sca'].result()),))
            train_metrics['obs_classif_sca'].reset_states()

            if config.use_validation:
                # Compute validation loss
                total_loss = 0
                num_batches_per_epoch = val_data.get_num_batches()
                for idx, batch in tqdm(val_data.get_batches(config.batch_size, num_steps = num_batches_per_epoch), total = num_batches_per_epoch, ascii = True):
                    # Format the data
                    batch_inputs, batch_targets = get_batch(batch, config)
                    batch_loss                  = self.batch_step(batch_inputs,batch_targets, val_metrics, training=False)
                    total_loss+= batch_loss
                # End epoch
                total_loss = total_loss / num_batches_per_epoch
                print('[TRN] Epoch {}. Validation loss {:.4f}'.format(epoch + 1, total_loss ))
                val_loss_results.append(total_loss)
                # Evaluate ADE, FDE metrics on validation data
                val_quantitative_metrics = self.quantitative_evaluation(val_data,config)
                val_metrics_results['ade'].append(val_quantitative_metrics['ade'])
                val_metrics_results['fde'].append(val_quantitative_metrics['fde'])
                if val_quantitative_metrics["ade"]< best['ade']:
                    best['ade'] = val_quantitative_metrics["ade"]
                    best['fde'] = val_quantitative_metrics["fde"]
                    best["patchId"]= idx
                    # Save the best model so far
                    checkpoint.write(checkpoint_prefix+'-best')
                print('[TRN] Epoch {}. Validation ADE {:.4f}'.format(epoch + 1, val_quantitative_metrics['ade']))

        # Training the classifier
        for epoch in range(10):
            print('Epoch {}.'.format(epoch + 1))
            # Cycle over batches
            total_loss = 0
            num_batches_per_epoch = train_data.get_num_batches()
            for idx, batch in tqdm(train_data.get_batches(config.batch_size, num_steps = num_batches_per_epoch, shuffle=True), total = num_batches_per_epoch, ascii = True):
                # Format the data
                batch_inputs, batch_targets = get_batch(batch, config)
                # Run the forward pass of the layer.
                # Compute the loss value for this minibatch.
                batch_loss = self.batch_step(batch_inputs, batch_targets, train_metrics, training=True)
                total_loss+= batch_loss
            # End epoch
            total_loss = total_loss / num_batches_per_epoch
            train_loss_results.append(total_loss)

            # Display information about the current state of the training loop
            print('[TRN] Epoch {}. Training loss {:.4f}'.format(epoch + 1, total_loss ))
            print('[TRN] Training accuracy of classifier p(z|x)   {:.4f}'.format(float(train_metrics['obs_classif_sca'].result()),))
            train_metrics['obs_classif_sca'].reset_states()

        return train_loss_results,val_loss_results,val_metrics_results,best["patchId"]

    def quantitative_evaluation(self,test_data,config):
        l2dis = []
        num_batches_per_epoch = test_data.get_num_batches()
        for idx, batch in tqdm(test_data.get_batches(config.batch_size, num_steps = num_batches_per_epoch, shuffle=True), total = num_batches_per_epoch, ascii = True):
            # Format the data
            batch_inputs, batch_targets = get_batch(batch, config)
            pred_out,__                 = self.batch_predict(batch_inputs,batch_targets.shape[1],1)
            pred_out                    = pred_out[0][0]
            this_actual_batch_size      = batch["original_batch_size"]
            d = []
            # For all the trajectories in the batch
            for i, (obs_traj_gt, pred_traj_gt) in enumerate(zip(batch["obs_traj"], batch["pred_traj"])):
                if i >= this_actual_batch_size:
                    break
                # TODO: replace
                normin = 1000.0
                diffmin= None
                for k in range(self.output_samples):
                    # Conserve the x,y coordinates of the kth trajectory
                    this_pred_out     = pred_out[k][i][:, :2] #[pred,2]
                    # Convert it to absolute (starting from the last observed position)
                    if config.output_representation=='dxdy':
                        this_pred_out_abs = relative_to_abs(this_pred_out, obs_traj_gt[-1])
                    else:
                        this_pred_out_abs = vw_to_abs(this_pred_out, obs_traj_gt[-1])
                    # Check shape is OK
                    assert this_pred_out_abs.shape == this_pred_out.shape, (this_pred_out_abs.shape, this_pred_out.shape)
                    # Error for ade/fde
                    diff = pred_traj_gt - this_pred_out_abs
                    diff = diff**2
                    diff = np.sqrt(np.sum(diff, axis=1))
                    # To keep the min
                    if tf.norm(diff)<normin:
                        normin  = tf.norm(diff)
                        diffmin = diff
                d.append(diffmin)
            l2dis += d
        ade = [t for o in l2dis for t in o] # average displacement
        fde = [o[-1] for o in l2dis] # final displacement
        return { "ade": np.mean(ade), "fde": np.mean(fde)}

    # Perform a qualitative evaluation over a batch of n_trajectories
    def qualitative_evaluation(self,batch,config,background=None,homography=None,flip=False,n_peds_max=1000,display_mode=None):
        traj_obs      = []
        traj_gt       = []
        traj_pred     = []
        neighbors     = []
        distributions = []
        batch_inputs, batch_targets = get_batch(batch, config)
        # Perform prediction
        if config.is_mc_dropout:
             mc_samples, mc_probabilities = self.batch_predict(batch_inputs,batch_targets.shape[1],config.mc_samples)
        else:
             mc_samples, mc_probabilities = self.batch_predict(batch_inputs,batch_targets.shape[1])

        # Cycle over the trajectories
        for i, (obs_traj_gt, pred_traj_gt, neighbors_gt) in enumerate(zip(batch["obs_traj"], batch["pred_traj"], batch["obs_neighbors"])):
            if i>=n_peds_max:
                break
            this_pred_out_abs_set = []
            for l in range(len(mc_samples)):
                pred_traj, pred_att_weights = mc_samples[l]
                mc_pred_set = []
                for k in range(self.output_samples):
                    # Conserve the x,y coordinates
                    if (pred_traj[k][i].shape[0]==config.pred_len):
                        this_pred_out     = pred_traj[k][i][:, :2]
                        # Convert it to absolute (starting from the last observed position)
                        if config.output_representation=='dxdy':
                            this_pred_out_abs = relative_to_abs(this_pred_out, obs_traj_gt[-1])
                        else:
                            this_pred_out_abs = vw_to_abs(this_pred_out, obs_traj_gt[-1])
                        mc_pred_set.append(this_pred_out_abs)
                mc_pred_set = tf.stack(mc_pred_set,axis=0)
                this_pred_out_abs_set.append(mc_pred_set)
            this_pred_out_abs_set = tf.stack(this_pred_out_abs_set,axis=0)
            # Keep all the trajectories
            traj_obs.append(obs_traj_gt)
            traj_gt.append(pred_traj_gt)
            traj_pred.append(this_pred_out_abs_set)
            neighbors.append(neighbors_gt)
        # Plot ground truth and predictions
        plot_gt_preds(traj_gt,traj_obs,traj_pred,neighbors,mc_probabilities,background,homography,flip=flip,display_mode=display_mode)
