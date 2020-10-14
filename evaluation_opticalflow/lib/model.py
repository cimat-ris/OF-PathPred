import functools
import operator
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

class Model_Parameters(object):
    """Model parameters.
    """
    def __init__(self, train_num_examples, add_kp = False, add_social = False):
        # -----------------
        # Observation/prediction lengths
        self.obs_len  = 8
        self.pred_len = 12

        self.add_kp             = add_kp
        self.train_num_examples = train_num_examples
        self.add_social         = add_social
        # Key points
        self.kp_size = 18
        # optical flow
        self.flow_size = 64
        # For training
        self.num_epochs = 30
        self.batch_size = 20 # batch size
        self.validate   = 300
        # Network architecture
        self.P               = 2 # Dimension
        self.enc_hidden_size = 64 # el nombre lo dice
        self.dec_hidden_size = 64
        self.emb_size        = 64
        self.keep_prob       = 0.7 # dropout

        self.min_ped      = 1
        self.seq_len      = self.obs_len + self.pred_len

        self.activation_func  = tf.nn.tanh
        self.activation_func1 = tf.nn.relu
        self.multi_decoder = False
        self.modelname = 'gphuctl'

        self.init_lr = 0.002 # 0.01
        self.learning_rate_decay = 0.85
        self.num_epoch_per_decay = 2.0
        self.optimizer = 'adam'
        self.emb_lr = 1.0
        # To save the best model
        self.load_best = True

class TrajectoryEncoder(layers.Layer):
    def __init__(self, config):
        # xy encoder: [N,T1,h_dim]
        super(TrajectoryEncoder, self).__init__(name="traj_enc")
        # Linear embedding of the observed trajectories (for each x,y)
        self.traj_xy_emb_enc = tf.keras.layers.Dense(config.emb_size,
            activation=config.activation_func,
            name='traj_enc_emb')
        # Dropout
        self.dropout = tf.keras.layers.Dropout(1.0-config.keep_prob)
        # LSTM cell
        self.lstm_cell = tf.keras.layers.LSTMCell(config.enc_hidden_size,
            name   = 'traj_enc_cell',
            dropout= 1.0-config.keep_prob,
            recurrent_dropout=1.0-config.keep_prob)
        # Recurrent neural network using the previous cell
        # Initial state is zero
        self.lstm      = tf.keras.layers.RNN(self.lstm_cell,
            return_sequences=True,
            return_state=True)

    def call(self,traj_inputs):
        # Linear embedding of the observed trajectories
        x = self.traj_xy_emb_enc(traj_inputs)
        # Dropout
        x = self.dropout(x)
        # Applies the position sequence through the LSTM
        return self.lstm(x)

class SocialEncoder(layers.Layer):
    def __init__(self, config):
        super(SocialEncoder, self).__init__(name="social_encoder")
        # Linear embedding of the social part
        self.traj_social_emb_enc = tf.keras.layers.Dense(config.emb_size,
            activation=config.activation_func,
            name='social_enc_emb')
        # Dropout
        self.dropout = tf.keras.layers.Dropout(1.0-config.keep_prob)
        # LSTM cell
        self.lstm_cell = tf.keras.layers.LSTMCell(config.enc_hidden_size,
            name   = 'social_enc_cell',
            dropout= 1.0-config.keep_prob,
            recurrent_dropout=1.0-config.keep_prob)
        # Recurrent neural network using the previous cell
        self.lstm      = tf.keras.layers.RNN(self.lstm_cell,
            return_sequences=True,
            return_state=True)

    def call(self,social_inputs):
        # Linear embedding of the observed trajectories
        x = self.traj_social_emb_enc(social_inputs)
        # Dropout
        x = self.dropout(x)
        # Applies the position sequence through the LSTM
        return self.lstm(x)

class TrajectoryDecoder(layers.Layer):
    def __init__(self, config):
        super(TrajectoryDecoder, self).__init__(name="traj_dec")
        #if config.multi_decoder: # Multiple output mode
        #    self.dec_cell_traj = [tf.keras.layers.LSTMCell.LSTMCell(
        #        config.dec_hidden_size,
        #        dropout= 1.0-config.keep_prob,
        #        recurrent_dropout=1.0-config.keep_prob,
        #        name='traj_dec_%s' % i)
        #        for i in range(len(config.traj_cats))]
        #else: # Simple mode: LSTM, with hidden size config.dec_hidden_size
        self.dec_cell_traj = tf.keras.layers.LSTMCell(config.dec_hidden_size,
            dropout= 1.0-config.keep_prob,
            recurrent_dropout=1.0-config.keep_prob,
            name='traj_dec')
        self.recurrentLayer = tf.keras.layers.RNN(self.dec_cell_traj,return_sequences=True)

        # Linear embedding of the observed trajectories
        self.traj_xy_emb_dec = tf.keras.layers.Dense(config.emb_size,
            activation=config.activation_func,
            name='traj_enc_emb')

        # Mapping from h to positions
        self.h_to_xy = tf.keras.layers.Dense(config.P,
            activation=tf.identity,
            name='h_to_xy')

    # Call to the decoder
    def call(self, first_input, enc_last_state, enc_h, decoder_inputs):
        print("Inputs of the decoder")
        print(first_input.shape)
        print(enc_last_state.shape)
        print(enc_h.shape)
        print(decoder_inputs.shape)
        print("--------------")
        # Decoder inputs: gound truth trajectory (training)
        # T_pred = tf.shape(decoder_inputs)[1]  # Value of T2 (prediction length)
        # Embedding
        decoder_inputs_emb = self.traj_xy_emb_dec(decoder_inputs)
        # Application of the RNN
        print("Applying Rnn")
        decoder_out_h = self.recurrentLayer(decoder_inputs_emb)
        print(decoder_out_h.shape)
        # [T2,N,dec_hidden_size]
        # decoder_out_h        = decoder_out_ta.stack()
        # [N,T2,dec_hidden_size]
        # Mapping to positions
        # decoder_out   = self.hidden2xy(decoder_out_h)
        decoder_out   = self.h_to_xy(decoder_out_h)
        return decoder_out


# The model
class TrajectoryEncoderDecoder(models.Model):
    def __init__(self, config):
        super(TrajectoryEncoderDecoder, self).__init__(name="traj_encoder_decoder")
        self.traj_enc     = TrajectoryEncoder(config)
        #self.soc_enc      = SocialEncoder(config)
        self.traj_dec     = TrajectoryDecoder(config)
        self.add_social   = config.add_social
        self.multi_decoder= config.multi_decoder

    def call(self,inputs):
        traj_inputs  = inputs[0]
        print(traj_inputs.shape)
        traj_pred_gt = inputs[1]
        # ----------------------------------------------------------
        # the obs part is the same for training and testing
        # obs_out is only used in training
        # encoder, decoder
        # top_scope is used for variable inside
        # encode and decode if want to share variable across
        # xy encoder: [N,T1,h_dim]

        # Applies the position sequence through the LSTM
        traj_obs_enc_h, traj_obs_enc_last_state, __ = self.traj_enc(traj_inputs)
        # Get the hidden states and the last hidden state, separately, and add them to the lists
        enc_h_list          = [traj_obs_enc_h]
        enc_last_state_list = [traj_obs_enc_last_state]

        # Social interaccion (through optical flow)
        # if self.add_social and soc_inputs is not None:
            # Applies the person pose (keypoints) sequence through the LSTM
        #     soc_obs_enc_h, soc_obs_enc_last_state, __ = self.soc_enc(soc_inputs)
        #    # Get hidden states and the last hidden state, separately, and add them to the lists
        #    enc_h_list.append(soc_obs_enc_h)
        #    enc_last_state_list.append(soc_obs_enc_last_state)

        # Pack all observed hidden states (lists) from all M features into a tensor
        # The final size should be [N,M,T_obs,h_dim]
        obs_enc_h          = tf.stack(enc_h_list, axis=1)

        # Concatenate last states (in the list) from all M features into a tensor
        # The final size should be [N,M,h_dim]
        obs_enc_last_state = tf.stack(enc_last_state_list, axis=1)

        # ----------------------------- xy decoder-----------------------------------------
        # Last observed position from the trajectory
        traj_obs_last = traj_inputs[:, -1]

        # Multiple decoder
        if self.multi_decoder:
            # [N, num_traj_cat] # each is num_traj_cat classification
            # TODO
            pass
        else:
            # Single decoder called: takes the last observed position, the last encoding state,
            # the tensor of all hidden states, the number of prediction steps
            traj_pred_out = self.traj_dec(traj_obs_last,traj_obs_enc_last_state,obs_enc_h,traj_pred_gt)
        return traj_pred_out
