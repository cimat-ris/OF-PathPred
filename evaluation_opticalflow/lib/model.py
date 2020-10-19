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
    def __init__(self):
        super(FocalAttention, self).__init__(name="focal_attention")

    def call(self,query, context):
        # query  : [N,D1]
        # context: [N,M,T,D2]
        print("*** focal attention ***")
        # Tensor dimensions
        _, D1       = query.get_shape().as_list()
        _, K, T, D2 = context.get_shape().as_list()
        assert d == d2
        # [N,d] -> [N,K,T,d]
        query_aug = tf.tile(tf.expand_dims(tf.expand_dims(query, 1), 1), [1, K, T, 1])
        # cosine simi
        query_aug_norm = tf.nn.l2_normalize(query_aug, -1)
        context_norm   = tf.nn.l2_normalize(context, -1)
        # [N, K, T]
        a_logits = tf.reduce_sum(tf.multiply(query_aug_norm, context_norm), 3)
        a_logits_maxed = tf.reduce_max(a_logits, 2)  # [N,K]
        print(a_logits_maxed)
        return a_logits_maxed
        #attended_context = softsel(softsel(context, a_logits), a_logits_maxed)
        #return attended_context

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
"""
class TrajectoryDecoder(layers.Layer):
    def __init__(self, config):
        super(TrajectoryDecoder, self).__init__(name="traj_dec")
        # TODO: multiple decoder to be done
        #if config.multi_decoder: # Multiple output mode
        #    self.dec_cell_traj = [tf.keras.layers.LSTMCell.LSTMCell(
        #        config.dec_hidden_size,
        #        dropout= 1.0-config.keep_prob,
        #        recurrent_dropout=1.0-config.keep_prob,
        #        name='traj_dec_%s' % i)
        #        for i in range(len(config.traj_cats))]
        #else: # Simple mode: LSTM, with hidden size config.dec_hidden_size
        #self.dec_cell_traj = tf.keras.layers.LSTMCell(config.dec_hidden_size,
        #    dropout= 1.0-config.keep_prob,
        #    recurrent_dropout=1.0-config.keep_prob,
        #    name='traj_dec')
        #self.dec_cell_traj  = DecoderLSTMCell(config.dec_hidden_size,
        #    dropout= 1.0-config.keep_prob,
        #    recurrent_dropout=1.0-config.keep_prob,
        #    name='traj_dec')
        self.dec_cell_traj  = tf.keras.layers.LSTMCell(config.dec_hidden_size,
            dropout= 1.0-config.keep_prob,
            recurrent_dropout=1.0-config.keep_prob,
            name='traj_dec')
        self.recurrentLayer = tf.keras.layers.RNN(self.dec_cell_traj,return_sequences=True)

        # Linear embedding of the observed trajectories
        self.traj_xy_emb_dec = tf.keras.layers.Dense(config.emb_size,
            activation=config.activation_func,
            name='traj_enc_emb')

        # Attention layer
        self.focal_attention = FocalAttention()

        # Mapping from h to positions
        self.h_to_xy = tf.keras.layers.Dense(config.P,
            activation=tf.identity,
            name='h_to_xy')

    # Call to the decoder
    def call(self, first_input, enc_last_state, enc_h, decoder_inputs):
        # Decoder inputs: gound truth trajectory (training)
        # T_pred = tf.shape(decoder_inputs)[1]  # Value of T2 (prediction length)
        # Embedding
        decoder_inputs_emb = self.traj_xy_emb_dec(decoder_inputs)
        # Application of the RNN: [N,T2,dec_hidden_size]
        decoder_out_h = self.recurrentLayer(decoder_inputs_emb)
        # Mapping to positions
        decoder_out   = self.h_to_xy(decoder_out_h)
         # Attention
        # [N,h_dim]
        # query is next_cell_state.h
        # context is enc_h
        # attended_encode_states = self.focal_attention(next_cell_state.h, enc_h)
        # Concatenate previous xy embedding, attended encoded states
        # [N,emb+h_dim]
        # next_input = tf.concat([xy_emb, attended_encode_states], axis=1)
        return decoder_out


# The model
class TrajectoryEncoderDecoder(models.Model):
    def __init__(self,config,input_shape):
        super(TrajectoryEncoderDecoder, self).__init__(name="traj_encoder_decoder")
        self.add_social   = config.add_social
        self.multi_decoder= config.multi_decoder
        # Input layers
        self.input_layer1 = layers.Input(input_shape[0],)
        self.input_layer2 = layers.Input(input_shape[1])
        # Encoding: Positions
        self.traj_enc     = TrajectoryEncoder(config)
        self.traj_dec     = TrajectoryDecoder(config)
        if (self.add_social):
            # In the case of handling social interactions, add a third input
            self.input_layer3 = layers.Input(input_shape[2])
            # Encoding: Social interactions
            self.soc_enc      = SocialEncoder(config)
            # Get output layer now with `call` method
            self.out = self.call([self.input_layer1,self.input_layer2,self.input_layer3])
        else:
            # Get output layer now with `call` method
            self.out = self.call([self.input_layer1,self.input_layer2])
        # Call init again. This is a workaround for being able to use summary
        super(TrajectoryEncoderDecoder, self).__init__(
            inputs=tf.cond(self.add_social, lambda: [self.input_layer1,self.input_layer2,self.input_layer3], lambda: [self.input_layer1,self.input_layer2]),
            outputs=self.out)


    def call(self,inputs,training=False):
        # inputs[0] is the observed part
        # inputs[1] is the ground truth continuation, it is only used in training
        traj_obs_inputs  = inputs[0]
        traj_pred_gt     = inputs[1]
        if self.add_social:
            # inputs[2] are the social interaction features
            soc_inputs     = inputs[2]

        # ----------------------------------------------------------
        # Encoding
        # ----------------------------------------------------------
        # Applies the position sequence through the LSTM: [N,T1,H]
        traj_obs_enc_h, traj_obs_enc_last_state, __ = self.traj_enc(traj_obs_inputs)
        # Get the hidden states and the last hidden state,
        # separately, and add them to the lists
        enc_h_list          = [traj_obs_enc_h]
        enc_last_state_list = [traj_obs_enc_last_state]

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

        # ----------------------------- xy decoder-----------------------------------------
        # Last observed position from the trajectory
        traj_obs_last = traj_obs_inputs[:, -1]

        # TODO
        # Multiple decoder
        #if self.multi_decoder:
            # [N, num_traj_cat] # each is num_traj_cat classification
        #    pass
        #else:
        # Single decoder called: takes the last observed position, the last encoding state,
        # the tensor of all hidden states, the number of prediction steps
        traj_pred_out = self.traj_dec(traj_obs_last,traj_obs_enc_last_state,obs_enc_h,traj_pred_gt)
        return traj_pred_out

#def softmax(logits, scope=None):
#    """a flatten and reconstruct version of softmax."""
#    flat_logits = flatten(logits, 1)
#    flat_out = tf.nn.softmax(flat_logits)
#    out = reconstruct(flat_out, logits, 1)
#    return out

#def softsel(target, logits, use_sigmoid=False, scope=None):
#    """Apply attention weights."""
#    a = softmax(logits)  # shape is the same
#    target_rank = len(target.get_shape().as_list())
#    # [N,M,JX,JQ,2d] elem* [N,M,JX,JQ,1]
#    # second last dim
#    return tf.reduce_sum(tf.expand_dims(a, -1)*target, target_rank-2)
