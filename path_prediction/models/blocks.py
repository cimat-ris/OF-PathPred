import functools, os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K
from .modules import TrajectoryEncoder, SocialEncoder, FocalAttention, ObservedTrajectoryClassifier


"""
Trajectory decoder initializer.
Allows to generate multiple ouputs. By learning to fit variations on the initial states of the decoder.
"""
class TrajectoryDecoderInitializer(tf.keras.Model):
    def __init__(self, config, add_social=False):
        super(TrajectoryDecoderInitializer, self).__init__(name="trajectory_decoder_initializer")
        self.add_social     = add_social
        self.output_var_dirs= config.output_var_dirs
        # Dropout layer
        self.dropout        = layers.Dropout(config.dropout_rate)
        # Linear embeddings from trajectory to hidden state
        self.traj_enc_h_to_dec_h = [layers.Dense(config.dec_hidden_size,
            activation=tf.keras.activations.relu,
            name='traj_enc_h_to_dec_h_%s'%i)  for i in range(self.output_var_dirs)]
        self.traj_enc_c_to_dec_c = [layers.Dense(config.dec_hidden_size,
            activation=tf.keras.activations.relu,
            name='traj_enc_c_to_dec_c_%s'%i)  for i in range(self.output_var_dirs)]
        if self.add_social:
            # Linear embeddings from social state to hidden state
            self.traj_soc_h_to_dec_h = [layers.Dense(config.dec_hidden_size,
                activation=tf.keras.activations.relu,
                name='traj_soc_h_to_dec_h_%s'%i)  for i in range(self.output_var_dirs)]
            self.traj_soc_c_to_dec_c = [layers.Dense(config.dec_hidden_size,
                activation=tf.keras.activations.relu,
                name='traj_soc_c_to_dec_c_%s'%i)  for i in range(self.output_var_dirs)]

    # Call to the decoder initializer
    def __call__(self, encoders_states, training=None):
        # The list of decoder states in decoder_init_states
        decoder_init_states = []
        traj_encoder_states  = encoders_states[0]
        # Append this pair of hidden states to the list of hypothesis (mean value)
        decoder_init_states.append(traj_encoder_states)
        if self.add_social:
            soc_encoder_states = encoders_states[1]
        for i in range(self.output_var_dirs):
            # Map the trajectory hidden states to variations of the initializer state
            decoder_init_dh  = self.traj_enc_h_to_dec_h[i](traj_encoder_states[0])
            decoder_init_dc  = self.traj_enc_c_to_dec_c[i](traj_encoder_states[1])
            if self.add_social:
                # Map the social features hidden states to variations of the initializer state
                decoder_init_dh = decoder_init_dh + self.traj_soc_h_to_dec_h[i](soc_encoder_states[0])
                decoder_init_dc = decoder_init_dc + self.traj_soc_c_to_dec_c[i](soc_encoder_states[1])
            # Define two opposite states based on these variations
            decoder_init_h   = traj_encoder_states[0]+decoder_init_dh
            decoder_init_c   = traj_encoder_states[1]+decoder_init_dc
            decoder_init_states.append([decoder_init_h,decoder_init_c])
            decoder_init_h   = traj_encoder_states[0]-decoder_init_dh
            decoder_init_c   = traj_encoder_states[1]-decoder_init_dc
            decoder_init_states.append([decoder_init_h,decoder_init_c])
        return decoder_init_states

################################################################################
############# Encoding
################################################################################
""" Custom model class for the encoding part (trajectory and social context)
"""
class TrajectoryAndContextEncoder(tf.keras.Model):
    def __init__(self,config):
        super(TrajectoryAndContextEncoder, self).__init__()
        # Flag for using social features
        self.add_social     = config.add_social
        # The RNN stack size
        self.stack_rnn_size = config.stack_rnn_size
        # Input layers
        obs_shape  = (config.obs_len,config.P)
        soc_shape  = (config.obs_len,config.flow_size)
        self.input_layer_traj = layers.Input(obs_shape,name="observed_trajectory")
        # Encoding: Positions
        self.traj_enc         = TrajectoryEncoder(config)
        # Classifier
        self.obs_classif      = ObservedTrajectoryClassifier(config)

        # We use the social features only when the two flags (add_social and add_attention are on)
        if (self.add_social):
            # In the case of handling social interactions, add a third input
            self.input_layer_social = layers.Input(soc_shape,name="social_features")
            # Encoding: Social interactions
            self.soc_enc            = SocialEncoder(config)
            # Get output layer now with `call` method
            self.out = self.call([self.input_layer_traj,self.input_layer_social])
        else:
            # Get output layer now with `call` method
            self.out = self.call([self.input_layer_traj])

        # Call init again. This is a workaround for being able to use summary()
        super(TrajectoryAndContextEncoder, self).__init__(
            inputs=tf.cond(self.add_social, lambda: [self.input_layer_traj,self.input_layer_social], lambda: [self.input_layer_traj]),
            outputs=self.out,name="trajectory_context_encoder")

    # Call expects an **array of inputs**
    def call(self,inputs,training=None):
        # inputs[0] is the observed trajectory part
        traj_obs_inputs  = inputs[0]
        if self.add_social:
            # inputs[1] are the social interaction features
            soc_inputs     = inputs[1]
            tf.debugging.assert_all_finite(soc_inputs,"PBM")

        # ----------------------------------------------------------
        # Encoding
        # ----------------------------------------------------------
        # Applies the position sequence through the LSTM: [N,T1,H]
        # In the case of stacked cells, output is:
        # sequence of outputs , last states (h,c) level 1, last states (h,c) level 2, ...
        outputs          = self.traj_enc(traj_obs_inputs,training=training)
        # Sequence of outputs at the highst level
        traj_h_seq       = outputs[0]
        # The last pairs of states, for each level of the stackd RNN
        traj_last_states = outputs[1:1+self.stack_rnn_size]
        # Get the sequence of output hidden states into enc_h_list
        enc_h_list          = [traj_h_seq]
        # ----------------------------------------------------------
        # Social interaccion (through optical flow)
        # ----------------------------------------------------------
        if self.add_social:
            # Applies the optical flow descriptor through the LSTM
            outputs = self.soc_enc(soc_inputs,training=training)
            # Last states from social encoding
            soc_last_states = [outputs[1],outputs[2]]
            # Sequences of outputs from the social encoding
            soc_h_seq       = outputs[0]
            # Get soc_h_seq into to the list enc_h_list
            enc_h_list.append(soc_h_seq)
        # Pack all observed hidden states (lists) from all M features into a context tensor
        # The final size should be [N,M,T_obs,h_dim]
        context          = tf.stack(enc_h_list, axis=1)
        # Apply classifier to guess what is th most probable output
        obs_classif_logits = self.obs_classif(traj_last_states[0][0])
        if self.add_social:
            #return traj_last_states,soc_last_states, obs_enc_h
            return [traj_last_states[0],soc_last_states], context, obs_classif_logits
        else:
            #return traj_last_states, obs_enc_h
            return [traj_last_states[0]], context, obs_classif_logits
