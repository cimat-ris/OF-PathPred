import tensorflow as tf
import os,logging,operator,functools,sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, losses

"""
Very Basic RNN-based Predictor.
"""
class PredictorDetRNN(keras.Model):
    """
    Basic Model parameters.
    """
    class parameters(object):
        def __init__(self, rnn_type='gru'):
            # -----------------
            # Observation/prediction lengths
            self.obs_len        = 8
            self.pred_len       = 12
            self.seq_len        = self.obs_len + self.pred_len
            # Coordinates
            self.coords_mode    = "rel_rot"
            # Rnn type
            self.rnn_type       = rnn_type
            # For training
            self.num_epochs     = 15
            self.batch_size     = 256
            self.use_validation = True
            # Network architecture
            self.P              =   2 # Dimensions of the position vectors
            self.rnn_type       = rnn_type
            self.enc_hidden_size= 64   # Hidden size of the RNN encoder
            self.dec_hidden_size= self.enc_hidden_size # Hidden size of the RNN decoder
            self.emb_size       = 64  # Embedding size
            self.dropout_rate   = 0.3  # Dropout rate during training
            self.activation_func= tf.nn.tanh
            self.optimizer      = 'adam'
            self.initial_lr     = 0.01
            self.weight_decay   = 0.005


    def __init__(self, config):
        super(PredictorDetRNN, self).__init__()
        self.rnn_type = config.rnn_type
        # Layers
        self.embedding = layers.Dense(config.emb_size, activation=config.activation_func)
        if self.rnn_type == "lstm":
            logging.info("Using LSTM RNNs")
            self.rnn   = layers.LSTM(config.enc_hidden_size, return_sequences=False, return_state=True,activation='tanh',dropout=config.dropout_rate)
        else:
            logging.info("Using GRU RNNs")            
            self.rnn   = layers.GRU(config.enc_hidden_size, return_sequences=False, return_state=True,activation='tanh',dropout=config.dropout_rate)
        self.dropout   = layers.Dropout(config.dropout_rate,name="dropout_decode")
        self.decoder   = layers.Dense(config.P, activation=tf.identity)
        # Loss function
        self.loss_fun  =  losses.LogCosh()

    def call(self, X, y, training=False):
        nbatches = len(X)
        # Last positions
        x_last = tf.reshape(X[:,-1,:], (nbatches, 1, -1))
        # Apply layers
        emb                = self.embedding(X) # Embedding
        if self.rnn_type == "lstm":
            lstm_out, hn1, cn1 = self.rnn(emb) # LSTM for batch [seq_len, batch_size, input_size]
        else:
            lstm_out, cn1 = self.rnn(emb) # GRU for batch [seq_len, batch_size, input_size]
        # Generate loss
        loss = 0
        # For each predicted timestep
        for i, target in enumerate(tf.transpose(y, perm=[1, 0, 2])):
            emb_last           = self.embedding(x_last)
            if self.rnn_type == "lstm":
                lstm_out, hn2, cn2 = self.rnn(emb_last, initial_state=[hn1, cn1]) # LSTM
            else:
                lstm_out, cn2 = self.rnn(emb_last, initial_state=[cn1]) # GRU
            cn2  =self.dropout(cn2)
            # Decoder and Prediction
            dec  = self.decoder(cn2) # [batch_size, input_size]
            dec  = tf.expand_dims(dec, 1)
            t_pred = dec + x_last    # [batch_size, 1, input_size]
            # Calculate the loss
            loss += (y.shape[1]-i)*self.loss_fun(t_pred, tf.reshape(target, (len(target), 1, -1)))
            # Update the last position
            if training:
                x_last = tf.reshape(target, (len(target), 1, -1))
            else:
                x_last = t_pred
            if self.rnn_type == "lstm":
                hn1 = hn2
            cn1 = cn2
        return loss

    def predict(self, inputs, dim_pred= 1):
        traj_inputs = inputs[0]
        nbatches    = len(traj_inputs)
        # Last position traj
        x_last = tf.reshape(traj_inputs[:,-1,:], (nbatches, 1, -1))
        # Embedding and Recurrent Layers
        emb = self.embedding(traj_inputs) # encoder for batch
        if self.rnn_type == "lstm":
            lstm_out, hn1, cn1 = self.rnn(emb) # LSTM
        else:
            lstm_out, cn1 = self.rnn(emb) # GRU
        pred = []
        for i in range(dim_pred):
            emb_last = self.embedding(x_last) # Embedding
            if self.rnn_type == "lstm":
                lstm_out, hn2, cn2 = self.rnn(emb_last, initial_state=[hn1, cn1]) # LSTM
            else:
                lstm_out, cn2 = self.rnn(emb_last, initial_state=[cn1]) # GRU
            # Decoder and Prediction
            dec = self.decoder(cn2)
            dec = tf.expand_dims(dec, 1)
            t_pred = dec + x_last
            pred.append(t_pred)
            # Update the last position
            x_last = t_pred
            if self.rnn_type == "lstm":
                hn1 = hn2
            cn1 = cn2
        # Concatenate the predictions and return
        return tf.expand_dims(tf.concat(pred, 1),1)
