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
        # Optical flow
        self.flow_size = 64
        # For training
        self.num_epochs = 200
        self.batch_size = 32  # batch size
        self.validate   = 300
        # Network architecture
        self.P               = 2 # Dimension
        self.enc_hidden_size = 64 #
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

""" Trajectory encoder through embedding+RNN.
"""
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

""" Social encoding through embedding+RNN.
"""
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
        B         = self.reshape(tf.nn.softmax(self.flatten(S)))
        BQ        = tf.reduce_sum(tf.expand_dims(B, -1)*context,2)
        # Weigthts for features, maxed over time: [N,M]
        Sm        = tf.reduce_max(S, 2)
        A         = tf.nn.softmax(Sm)
        AQ        = tf.reduce_sum(tf.expand_dims(A, -1)*BQ,1)
        return tf.expand_dims(AQ,1)

""" Custom model class for the encoding part (trajectory and context)
"""
class TrajectoryAndContextEncoder(tf.keras.Model):
    def __init__(self,config):
        super(TrajectoryAndContextEncoder, self).__init__(name="traj_ctxt_enc")
        self.add_social   = config.add_social
        # Input layers
        obs_shape  = (config.obs_len,config.P)
        soc_shape  = (config.obs_len,config.flow_size)
        self.input_layer_traj = layers.Input(obs_shape)
        # Encoding: Positions
        self.traj_enc     = TrajectoryEncoder(config)
        if (self.add_social):
            # In the case of handling social interactions, add a third input
            self.input_layer_social = layers.Input(soc_shape)
            # Encoding: Social interactions
            self.soc_enc            = SocialEncoder(config)
            # Get output layer now with `call` method
            self.out = self.call([self.input_layer_traj,self.input_layer_social])
        else:
            # Get output layer now with `call` method
            self.out = self.call([self.input_layer_traj])
        # Call init again. This is a workaround for being able to use summary
        super(TrajectoryAndContextEncoder, self).__init__(
            inputs=tf.cond(self.add_social, lambda: [self.input_layer_traj,self.input_layer_social], lambda: [self.input_layer_traj]),
            outputs=self.out)

    def call(self,inputs,training=False):
        # inputs[0] is the observed part
        traj_obs_inputs  = inputs[0]
        if self.add_social:
            # inputs[1] are the social interaction features
            soc_inputs     = inputs[1]
        # ----------------------------------------------------------
        # Encoding
        # ----------------------------------------------------------
        # Applies the position sequence through the LSTM: [N,T1,H]
        traj_obs_enc_h, traj_obs_enc_last_state1, traj_obs_enc_last_state2 = self.traj_enc(traj_obs_inputs)
        # Get the hidden states and the last hidden state,
        # separately, and add them to the lists
        enc_h_list          = [traj_obs_enc_h]
        enc_last_state_list = [traj_obs_enc_last_state1]
        # ----------------------------------------------------------
        # Social interaccion (through optical flow)
        # ----------------------------------------------------------
        if self.add_social:
            # Applies the person pose (keypoints) sequence through the LSTM
            soc_obs_enc_h, soc_obs_enc_last_state, __ = self.soc_enc(soc_inputs)
            # Get hidden states and the last hidden state, separately, and add them to the lists
            enc_h_list.append(soc_obs_enc_h)
            enc_last_state_list.append(soc_obs_enc_last_state)
        # Pack all observed hidden states (lists) from all M features into a tensor
        # The final size should be [N,M,T_obs,h_dim]
        obs_enc_h          = tf.stack(enc_h_list, axis=1)
        # Concatenate last states (in the list) from all M features into a tensor
        # The final size should be [N,M,h_dim]
        obs_enc_last_state = tf.stack(enc_last_state_list, axis=1)
        return traj_obs_enc_last_state1,traj_obs_enc_last_state2,obs_enc_h

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
        #tf.print(inputs.shape, output_stream=sys.stderr)
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

""" Trajectory decoder.
    Generates the next position
"""
class TrajectoryDecoder(tf.keras.Model):
    def __init__(self, config):
        super(TrajectoryDecoder, self).__init__(name="traj_dec")
        self.add_social   = config.add_social
        # TODO: multiple decoder to be done
        # Linear embedding of the observed trajectories
        self.traj_xy_emb_dec = tf.keras.layers.Dense(config.emb_size,
            activation=config.activation_func,
            name='traj_enc_emb')
        # RNN cell
        self.dec_cell_traj  = tf.keras.layers.LSTMCell(config.dec_hidden_size,
            recurrent_initializer='glorot_uniform',
            dropout= 1.0-config.keep_prob,
            recurrent_dropout=1.0-config.keep_prob,
            name='traj_dec')
        self.recurrentLayer = tf.keras.layers.RNN(self.dec_cell_traj,return_sequences=True,return_state=True)
        M = 1
        if (self.add_social):
            M=M+1
        # Attention layer
        self.focal_attention = FocalAttention(config,M)
        # Mapping from h to positions
        self.h_to_xy = tf.keras.layers.Dense(config.P,
            activation=tf.identity,
            name='h_to_xy')
        # Input layers
        dec_input_shape      = (1,config.P)
        self.input_layer_pos = layers.Input(dec_input_shape)
        enc_last_state_shape = (config.dec_hidden_size)
        self.input_layer_hid1= layers.Input(enc_last_state_shape)
        self.input_layer_hid2= layers.Input(enc_last_state_shape)
        # [N,M,T1,h_dim]
        ctxt_shape = (M,config.obs_len,config.enc_hidden_size)
        self.input_layer_ctxt = layers.Input(ctxt_shape)
        self.out = self.call(self.input_layer_pos,self.input_layer_hid1,self.input_layer_hid2,self.input_layer_ctxt)
        # Call init again. This is a workaround for being able to use summary
        super(TrajectoryDecoder, self).__init__(
                    inputs= [self.input_layer_pos,self.input_layer_hid1,self.input_layer_hid2,self.input_layer_ctxt],
                    outputs=self.out)

    # Call to the decoder
    def call(self, dec_input, enc_last_state1, enc_last_state2, context, firstCall=False,training=False):
        # Decoder inputs: position
        # Embedding
        decoder_inputs_emb = self.traj_xy_emb_dec(dec_input)
         # Attention: [N,1,h_dim]
        # query is decoder_out_h: [N,h_dim]
        query           = enc_last_state1
        attention       = self.focal_attention(query, context)
        # Augmented imput: [N,1,h_dim+emb]
        augmented_inputs= tf.concat([decoder_inputs_emb, attention], axis=2)
        # Application of the RNN: [N,T2,dec_hidden_size]
        decoder_out        = self.recurrentLayer(augmented_inputs,initial_state=(enc_last_state1, enc_last_state2))
        decoder_out_h      = decoder_out[0]
        decoder_out_states1= decoder_out[1]
        decoder_out_states2= decoder_out[2]
        # Mapping to positions
        decoder_out_xy = self.h_to_xy(decoder_out_h)
        # Concatenate previous xy embedding, attended encoded states
        # [N,emb+h_dim]
        # next_input = tf.concat([xy_emb, attended_encode_states], axis=1)
        return decoder_out_xy, decoder_out_states1, decoder_out_states2


# The main class
class TrajectoryEncoderDecoder():
    def __init__(self,config):
        self.add_social   = config.add_social
        self.multi_decoder= config.multi_decoder
        # Encoding: Positions and context
        self.enc = TrajectoryAndContextEncoder(config)
        self.enc.summary()
        self.dec = TrajectoryDecoder(config)
        self.dec.summary()
        # Instantiate an optimizer to train the models.
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)


    def train_step(self, batch_inputs, batch_targets):
        traj_obs_inputs = batch_inputs[0]
        # Last observed position from the trajectory
        traj_obs_last = traj_obs_inputs[:, -1]
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        loss_value = 0
        # Loss function
        loss_fn = keras.losses.MeanSquaredError()
        with tf.GradientTape() as g:
            # Apply trajectory and context encoding
            traj_obs_enc_last_state1, traj_obs_enc_last_state2, obs_enc_h = self.enc(batch_inputs, training=True)
            # The first input to the decoder is the last observed position [Nx1xK]
            dec_input = tf.expand_dims(traj_obs_last, 1)
            # Teacher forcing - feeding the target as the next input
            for t in range(1, batch_targets.shape[1]):
                # ------------------------ xy decoder--------------------------------------
                # passing enc_output to the decoder
                t_pred, dec_hidden1, dec_hidden2 = self.dec(dec_input,traj_obs_enc_last_state1,traj_obs_enc_last_state2,obs_enc_h,training=True)
                t_target = tf.expand_dims(batch_targets[:, t], 1)
                # Loss for
                loss_value += loss_fn(t_target, t_pred)
                # Using teacher forcing [Nx1xK]
                dec_input = tf.expand_dims(batch_targets[:, t], 1)
                traj_obs_enc_last_state1 = dec_hidden1
                traj_obs_enc_last_state2 = dec_hidden2

        variables = self.enc.trainable_weights + self.dec.trainable_weights
        # Get the gradients
        grads = g.gradient(loss_value, variables)
        # Run one step of gradient descent
        self.optimizer.apply_gradients(zip(grads, variables))
        # Average loss over the predicted times
        batch_loss = (loss_value / int(batch_targets.shape[1]))
        return batch_loss

    def predict(self, batch_inputs, n_steps):
        traj_obs_inputs = batch_inputs[0]
        traj_pred       = []
        # Last observed position from the trajectory
        traj_obs_last = traj_obs_inputs[:, -1]
        # Apply trajectory and context encoding
        traj_obs_enc_last_state1, traj_obs_enc_last_state2, obs_enc_h = self.enc(batch_inputs, training=False)
        # The first input to the decoder is the last observed position [Nx1xK]
        dec_input = tf.expand_dims(traj_obs_last, 1)
        for t in range(1, n_steps):
            # ------------------------ xy decoder--------------------------------------
            # passing enc_output to the decoder
            t_pred, dec_hidden1, dec_hidden2 = self.dec(dec_input,traj_obs_enc_last_state1,traj_obs_enc_last_state2,obs_enc_h,training=True)
            # Next input is the last position
            dec_input = t_pred
            traj_pred.append(t_pred)
            traj_obs_enc_last_state1 = dec_hidden1
            traj_obs_enc_last_state2 = dec_hidden2
        return tf.squeeze(tf.stack(traj_pred, axis=1))
