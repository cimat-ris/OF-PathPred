import tensorflow as tf
import functools
import operator
import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense

"""
Trajectory encoder through embedding+RNN.
"""
class TrajectoryEncoder(layers.Layer):
    def __init__(self, config):
        self.stack_rnn_size  = config.stack_rnn_size
        self.is_mc_dropout   = config.is_mc_dropout
        # xy encoder: [N,T1,h_dim]
        super(TrajectoryEncoder, self).__init__(name="trajectory_encoder")
        # Linear embedding of the observed positions (for each x,y)
        self.traj_xy_emb_enc = tf.keras.layers.Dense(config.emb_size,
            activation=config.activation_func,
            use_bias=True,
            name='position_embedding')
        # LSTM cell, including dropout, with a stacked configuration.
        # Output is composed of:
        # - the sequence of h's along time, from the highest level only: h1,h2,...
        # - last pair of states (h,c) for the first layer
        # - last pair of states (h,c) for the second layer
        # - ... and so on
        self.lstm_cells= [tf.keras.layers.LSTMCell(config.enc_hidden_size,
                name   = 'trajectory_encoder_cell',
                dropout= config.dropout_rate,
                recurrent_dropout=config.dropout_rate) for _ in range(self.stack_rnn_size)]
        self.lstm_cell = tf.keras.layers.StackedRNNCells(self.lstm_cells)
        # Recurrent neural network using the previous cell
        # Initial state is zero; We return the full sequence of h's and the pair of last states
        self.lstm      = tf.keras.layers.RNN(self.lstm_cell,
                return_sequences= True,
                return_state    = True)

    def call(self,traj_inputs,training=None):
        # Linear embedding of the observed trajectories
        x = self.traj_xy_emb_enc(traj_inputs)
        # Applies the position sequence through the LSTM
        # The training parameter is important for dropout
        return self.lstm(x,training=(training or self.is_mc_dropout))

"""
Social encoding through embedding+RNN.
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

"""
Focal attention layer.
"""
# TODO: test other attention models?
# TODO: analysis of the attention results
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
