import functools, os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K

"""
Trajectory encoder through embedding+RNN.
"""
class TrajectoryEncoder(tf.Module):
    def __init__(self, config):
        self.stack_rnn_size  = config.stack_rnn_size
        # xy encoder: [N,T1,h_dim]
        super(TrajectoryEncoder, self).__init__(name="trajectory_encoder")
        # Linear embedding of the observed positions (for each x,y)
        self.traj_xy_emb_enc = layers.Dense(config.emb_size,
            activation=config.activation_func,
            use_bias=True,
            name='position_embedding')
        # LSTM cell, including dropout, with a stacked configuration.
        # Output is composed of:
        # - the sequence of h's along time, from the highest level only: h1,h2,...
        # - last pair of states (h,c) for the first layer
        # - last pair of states (h,c) for the second layer
        # - ... and so on
        self.lstm_cells= [layers.LSTMCell(config.enc_hidden_size,
                name   = 'trajectory_encoder_cell',
                dropout= config.dropout_rate,
                recurrent_dropout=config.dropout_rate) for _ in range(self.stack_rnn_size)]
        self.lstm_cell = layers.StackedRNNCells(self.lstm_cells)
        # Recurrent neural network using the previous cell
        # Initial state is zero; We return the full sequence of h's and the pair of last states
        self.lstm      = layers.RNN(self.lstm_cell,
                return_sequences= True,
                return_state    = True)

    def __call__(self,traj_inputs,training=None):
        # Linear embedding of the observed trajectories
        x = self.traj_xy_emb_enc(traj_inputs)
        # Applies the position sequence through the LSTM
        # The training parameter is important for dropout
        return self.lstm(x,training=training)

"""
Social encoding through embedding+RNN.
"""
class SocialEncoder(tf.Module):
    def __init__(self, config):
        super(SocialEncoder, self).__init__(name="social_encoder")
        # Linear embedding of the social part
        self.traj_social_emb_enc = layers.Dense(config.emb_size,
            activation=config.activation_func,
            name='social_feature_embedding')
        # LSTM cell, including dropout
        self.lstm_cell = layers.LSTMCell(config.enc_hidden_size,
            name   = 'social_encoder_cell',
            dropout= config.dropout_rate,
            recurrent_dropout= config.dropout_rate)
        # Recurrent neural network using the previous cell
        self.lstm      = layers.RNN(self.lstm_cell,
            return_sequences= True,
            return_state    = True)

    def __call__(self,social_inputs,training=None):
        # Linear embedding of the observed trajectories
        x = self.traj_social_emb_enc(social_inputs)
        # Applies the position sequence through the LSTM
        x = self.lstm(x,training=training)
        return x

"""
Focal attention layer.
"""
# TODO: test other attention models?
# TODO: analysis of the attention results
class FocalAttention(tf.Module):
    def __init__(self,config,M):
        super(FocalAttention, self).__init__(name="focal_attention")
        self.flatten  = layers.Flatten()
        self.reshape  = layers.Reshape((M, config.obs_len))

    def __call__(self,query, context):
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


"""
Observed trajectory classifier: during training, takes the observed trajectory and predict the class
"""
class ObservedTrajectoryClassifier(tf.Module):
    def __init__(self, config):
        super(ObservedTrajectoryClassifier, self).__init__(name="observed_trajectory_classification")
        self.output_var_dirs= config.output_var_dirs
        self.output_samples = 2*config.output_var_dirs+1
        input_observed_shape= (config.enc_hidden_size)
        self.input_observed = keras.Input(shape=input_observed_shape, name="observed_trajectory_h")
        self.dense_layer_observed = layers.Dense(64, activation="relu", name="observed_dense")
        self.classification_layer = layers.Dense(self.output_samples, activation="softmax", name="classication")

    # Call to the classifier p(z|x,y)
    def __call__(self, observed_trajectory_h, training=None):
        # Linear embedding of the observed trajectories
        x = self.dense_layer_observed(observed_trajectory_h)
        return self.classification_layer(x)
