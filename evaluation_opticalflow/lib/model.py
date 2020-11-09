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
from tensorflow.keras.layers import Dense, Bidirectional
from tensorflow.keras import models

class Model_Parameters(object):
    """Model parameters.
    """
    def __init__(self, add_attention=True, add_kp = False, add_social = False, output_representation='dxdy'):
        # -----------------
        # Observation/prediction lengths
        self.obs_len        = 8
        self.pred_len       = 12
        self.seq_len        = self.obs_len + self.pred_len
        self.add_kp         = add_kp
        self.add_social     = add_social
        self.add_attention  = add_attention
        self.add_bidirection= False
        self.add_stacked_rnn= True
        self.output_representation = output_representation
        # Key points
        self.kp_size        = 18
        # Optical flow
        self.flow_size      = 64
        # For training
        self.num_epochs     = 50
        self.batch_size     = 512  # batch size
        self.use_validation = True
        # Network architecture
        self.P              =   2 # Dimensions of the position vectors
        self.enc_hidden_size= 256 # Default value in NextP
        self.dec_hidden_size= 256 # Default value in NextP
        self.emb_size       = 128 # Default value in NextP
        self.dropout_rate   = 0.3 # Default value in NextP

        #self.activation_func= tf.nn.sigmoid
        self.activation_func= tf.nn.tanh
        #self.activation_func= tf.nn.relu
        self.multi_decoder  = False
        self.modelname      = 'gphuctl'
        self.optimizer      = 'adam'

""" Trajectory encoder through embedding+RNN.
"""
class TrajectoryEncoder(layers.Layer):
    def __init__(self, config):
        self.add_bidirection = config.add_bidirection
        # xy encoder: [N,T1,h_dim]
        super(TrajectoryEncoder, self).__init__(name="traj_enc")
        # Linear embedding of the observed trajectories (for each x,y)
        self.traj_xy_emb_enc = tf.keras.layers.Dense(config.emb_size,
            activation=config.activation_func,
            use_bias=True,
            name='traj_enc_emb')
        # LSTM cell, including dropout
        if config.add_stacked_rnn==False:
            self.lstm_cell = tf.keras.layers.LSTMCell(config.enc_hidden_size,
                name   = 'traj_enc_cell',
                dropout= config.dropout_rate,
                recurrent_dropout=config.dropout_rate)
        else:
            self.lstm_cells = [tf.keras.layers.LSTMCell(config.enc_hidden_size,
                name   = 'traj_enc_cell',
                dropout= config.dropout_rate,
                recurrent_dropout=config.dropout_rate) for _ in range(2)]
            self.lstm_cell = tf.keras.layers.StackedRNNCells(self.lstm_cells)

        # Recurrent neural network using the previous cell
        # Initial state is zero
        if self.add_bidirection:
            self.lstm      = Bidirectional(tf.keras.layers.RNN(self.lstm_cell,
                return_sequences=True,
                return_state=True),merge_mode='sum')
        else:
            self.lstm      = tf.keras.layers.RNN(self.lstm_cell,
                return_sequences=True,
                return_state=True)

    def call(self,traj_inputs,training=None):
        # Linear embedding of the observed trajectories
        x = self.traj_xy_emb_enc(traj_inputs)
        # Applies the position sequence through the LSTM
        return self.lstm(x,training=training)

""" Social encoding through embedding+RNN.
"""
class SocialEncoder(layers.Layer):
    def __init__(self, config):
        super(SocialEncoder, self).__init__(name="social_encoder")
        # Linear embedding of the social part
        self.traj_social_emb_enc = tf.keras.layers.Dense(config.emb_size,
            activation=config.activation_func,
            name='social_enc_emb')
        # LSTM cell, including dropout
        self.lstm_cell = tf.keras.layers.LSTMCell(config.enc_hidden_size,
            name   = 'social_enc_cell',
            dropout= config.dropout_rate,
            recurrent_dropout= config.dropout_rate
            )
        # Recurrent neural network using the previous cell
        self.lstm      = tf.keras.layers.RNN(self.lstm_cell,
            return_sequences=True,
            return_state=True)

    def call(self,social_inputs,training=None):
        # Linear embedding of the observed trajectories
        x = self.traj_social_emb_enc(social_inputs)
        # Applies the position sequence through the LSTM
        return self.lstm(x,training=training)

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
        self.add_social     = config.add_social
        self.add_attention  = config.add_attention
        self.add_bidirection=config.add_bidirection
        self.add_stacked_rnn= config.add_stacked_rnn
        # Input layers
        obs_shape  = (config.obs_len,config.P)
        soc_shape  = (config.obs_len,config.flow_size)
        self.input_layer_traj = layers.Input(obs_shape)
        # Encoding: Positions
        self.traj_enc     = TrajectoryEncoder(config)
        if (self.add_attention and self.add_social):
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
        if self.add_bidirection:
            traj_obs_enc_h, traj_obs_enc_last_state1, traj_obs_enc_last_state2,__,__ = self.traj_enc(traj_obs_inputs,training=training)
        else:
            # In the case of stacked cells, output is:
            # final encoding , last states (h,c) level 1, last states (h,c) level 2, ...
            traj_obs_enc_h, traj_obs_enc_last_state1, traj_obs_enc_last_state2 = self.traj_enc(traj_obs_inputs,training=training)

        # Get the hidden states and the last hidden state,
        # separately, and add them to the lists
        enc_h_list          = [traj_obs_enc_h]
        # ----------------------------------------------------------
        # Social interaccion (through optical flow)
        # ----------------------------------------------------------
        if self.add_attention and self.add_social:
            # Applies the person pose (keypoints) sequence through the LSTM
            soc_obs_enc_h, soc_obs_enc_last_state, __, = self.soc_enc(soc_inputs,training=training)
            # Get hidden states and the last hidden state, separately, and add them to the lists
            enc_h_list.append(soc_obs_enc_h)
        # Pack all observed hidden states (lists) from all M features into a tensor
        # The final size should be [N,M,T_obs,h_dim]
        obs_enc_h          = tf.stack(enc_h_list, axis=1)
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
        self.add_social     = config.add_social
        self.add_attention  = config.add_attention
        self.add_stacked_rnn= config.add_stacked_rnn
        # TODO: multiple decoder to be done
        # Linear embedding of the observed trajectories
        self.traj_xy_emb_dec = tf.keras.layers.Dense(config.emb_size,
            activation=config.activation_func,
            name='traj_enc_emb')
        # RNN cell
        self.dec_cell_traj  = tf.keras.layers.LSTMCell(config.dec_hidden_size,
            recurrent_initializer='glorot_uniform',
            name='traj_dec',
            dropout= config.dropout_rate,
            recurrent_dropout=config.dropout_rate
            )
        self.recurrentLayer = tf.keras.layers.RNN(self.dec_cell_traj,return_sequences=True,return_state=True)
        self.M = 1
        if (self.add_attention and self.add_social):
            self.M=self.M+1

        # Attention layer
        if (self.add_attention):
            self.focal_attention = FocalAttention(config,self.M)
        # Dropout
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)
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
        ctxt_shape = (self.M,config.obs_len,config.enc_hidden_size)
        self.input_layer_ctxt = layers.Input(ctxt_shape)
        self.out = self.call(self.input_layer_pos,self.input_layer_hid1,self.input_layer_hid2,self.input_layer_ctxt)
        # Call init again. This is a workaround for being able to use summary
        super(TrajectoryDecoder, self).__init__(
                    inputs= [self.input_layer_pos,self.input_layer_hid1,self.input_layer_hid2,self.input_layer_ctxt],
                    outputs=self.out)

    # Call to the decoder
    def call(self, dec_input, enc_last_state1, enc_last_state2, context, firstCall=False,training=None):
        # Decoder inputs: position
        # Embedding
        decoder_inputs_emb = self.traj_xy_emb_dec(dec_input)
        # Attention: [N,1,h_dim]
        # query is decoder_out_h: [N,h_dim]
        query           = enc_last_state1
        if self.add_attention:
            attention       = self.focal_attention(query, context)
            # Augmented input: [N,1,h_dim+emb]
            augmented_inputs= tf.concat([decoder_inputs_emb, attention], axis=2)
        else:
            augmented_inputs= decoder_inputs_emb
        # Application of the RNN: [N,T2,dec_hidden_size]
        decoder_out        = self.recurrentLayer(augmented_inputs,initial_state=(enc_last_state1, enc_last_state2),training=training)
        decoder_out_h      = decoder_out[0]
        decoder_out_states1= decoder_out[1]
        decoder_out_states2= decoder_out[2]
        # TODO: dropout here?
        decoder_out_h = self.dropout(decoder_out_h,training=training)
        # Mapping to positions
        decoder_out_xy = self.h_to_xy(decoder_out_h)
        # Concatenate previous xy embedding, attended encoded states
        # [N,emb+h_dim]
        return decoder_out_xy, decoder_out_states1, decoder_out_states2

# The main class
class TrajectoryEncoderDecoder():
    # Constructor
    def __init__(self,config):
        # Flags for considering social interations, multiple decoder
        self.add_social     = config.add_social
        self.multi_decoder  = config.multi_decoder
        self.add_stacked_rnn= config.add_stacked_rnn
        # Encoder: Positions and context
        self.enc = TrajectoryAndContextEncoder(config)
        self.enc.summary()
        # Decoder
        self.dec = TrajectoryDecoder(config)
        self.dec.summary()

        initial_learning_rate = 0.01
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate,
                decay_steps=100000,
                decay_rate=0.96,
                staircase=True)

        # Instantiate an optimizer to train the models.
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
        self.optimizer = tf.keras.optimizers.Adadelta(learning_rate=lr_schedule)

        # Instantiate the loss operator
        #self.loss_fn = keras.losses.MeanSquaredError()
        self.loss_fn = keras.losses.LogCosh()

    # Trick to reset th weights: We save them and reload them
    def save_tmp(self):
        self.enc.save_weights('tmp_enc.h5')
        self.dec.save_weights('tmp_dec.h5')
    def load_tmp(self):
        self.enc.load_weights('tmp_enc.h5')
        self.dec.load_weights('tmp_dec.h5')

    # Single training/testing step, for one batch
    def batch_step(self, batch_inputs, batch_targets, training=True):
        traj_obs_inputs = batch_inputs[0]
        # Last observed position from the trajectory
        traj_obs_last = traj_obs_inputs[:, -1]
        # Variables
        variables = self.enc.trainable_weights + self.dec.trainable_weights
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        loss_value = 0
        with tf.GradientTape() as g:
            # Apply trajectory and context encoding
            traj_obs_enc_last_state1, traj_obs_enc_last_state2, context = self.enc(batch_inputs, training=training)
            if self.add_stacked_rnn:
                traj_obs_enc_last_state1 = traj_obs_enc_last_state1[1]
                traj_obs_enc_last_state2 = traj_obs_enc_last_state2[1]

            # The first input to the decoder is the last observed position [Nx1xK]
            dec_input = tf.expand_dims(traj_obs_last, 1)
            # Teacher forcing - feeding the target as the next input
            for t in range(0, batch_targets.shape[1]):
                # ------------------------ xy decoder--------------------------------------
                # passing enc_output to the decoder
                t_pred, dec_hidden1, dec_hidden2 = self.dec(dec_input,traj_obs_enc_last_state1,traj_obs_enc_last_state2,context,training=training)
                t_target = tf.expand_dims(batch_targets[:, t], 1)
                # Loss
                loss_value += (batch_targets.shape[1]-t)*self.loss_fn(t_target, t_pred)
                if training==True:
                    # Using teacher forcing [Nx1xK]
                    dec_input = tf.expand_dims(batch_targets[:, t], 1)
                else:
                    # Next input is the last predicted position
                    dec_input = t_pred
                traj_obs_enc_last_state1 = dec_hidden1
                traj_obs_enc_last_state2 = dec_hidden2
            # L2 weight decay
            loss_value += tf.add_n([ tf.nn.l2_loss(v) for v in variables
                        if 'bias' not in v.name ]) * 0.0008
        if training==True:
            # Get the gradients
            grads = g.gradient(loss_value, variables)
            # Run one step of gradient descent
            self.optimizer.apply_gradients(zip(grads, variables))
        # Average loss over the predicted times
        batch_loss = (loss_value / int(batch_targets.shape[1]))
        return batch_loss

    # Prediction (testing) for one batch
    def batch_predict(self, batch_inputs, n_steps):
        traj_obs_inputs = batch_inputs[0]
        # List for the predictions
        traj_pred       = []
        # Last observed position from the trajectories
        traj_obs_last = traj_obs_inputs[:, -1]
        # Apply trajectory and context encoding
        traj_obs_enc_last_state1, traj_obs_enc_last_state2, context = self.enc(batch_inputs, training=False)
        if self.add_stacked_rnn:
            traj_obs_enc_last_state1 = traj_obs_enc_last_state1[1]
            traj_obs_enc_last_state2 = traj_obs_enc_last_state2[1]

        # The first input to the decoder is the last observed position [Nx1xK]
        dec_input = tf.expand_dims(traj_obs_last, 1)
        for t in range(0, n_steps):
            # ------------------------ xy decoder--------------------------------------
            # Passing enc_output to the decoder
            t_pred, dec_hidden1, dec_hidden2 = self.dec(dec_input,traj_obs_enc_last_state1,traj_obs_enc_last_state2,context,training=False)
            # Next input is the last predicted position
            dec_input = t_pred
            # Add it to the list of predictions
            traj_pred.append(t_pred)
            # Reuse the hidden states for the next step
            traj_obs_enc_last_state1 = dec_hidden1
            traj_obs_enc_last_state2 = dec_hidden2
        return tf.squeeze(tf.stack(traj_pred, axis=1))

    # Training loop
    def training_loop(self,train_data,val_data,config,checkpoint,checkpoint_prefix):
        num_batches_per_epoch= train_data.get_num_batches()
        train_loss_results   = []
        val_loss_results     = []
        val_metrics_results  = { "ade": [], "fde": []}
        best                 = {'ade':999999, 'fde':0, 'batchId':-1}
        # Epochs
        for epoch in range(config.num_epochs):
            print('Epoch {}.'.format(epoch + 1))
            # Cycle over batches
            total_loss = 0
            num_batches_per_epoch = train_data.get_num_batches()
            for idx, batch in tqdm(train_data.get_batches(config.batch_size, num_steps = num_batches_per_epoch, shuffle=True), total = num_batches_per_epoch, ascii = True):
                # Format the data
                batch_inputs, batch_targets = get_batch(batch, config)
                # Run the forward pass of the layer.
                # Compute the loss value for this minibatch.
                batch_loss = self.batch_step(batch_inputs, batch_targets,training=True)
                total_loss+= batch_loss
            # End epoch
            total_loss = total_loss / num_batches_per_epoch
            train_loss_results.append(total_loss)

                # Saving (checkpoint) the model every 2 epochs
            if (epoch + 1) % 2 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)
            print('Epoch {}. Training loss {:.4f}'.format(epoch + 1, total_loss ))

            if config.use_validation:
                # Compute validation loss
                total_loss = 0
                num_batches_per_epoch = val_data.get_num_batches()
                for idx, batch in tqdm(val_data.get_batches(config.batch_size, num_steps = num_batches_per_epoch), total = num_batches_per_epoch, ascii = True):
                    # Format the data
                    batch_inputs, batch_targets = get_batch(batch, config)
                    batch_loss                  = self.batch_step(batch_inputs,batch_targets,training=False)
                    total_loss+= batch_loss
                # End epoch
                total_loss = total_loss / num_batches_per_epoch
                print('Epoch {}. Validation loss {:.4f}'.format(epoch + 1, total_loss ))
                val_loss_results.append(total_loss)
                # Evaluat ADE, FDE metrics on validation data
                val_metrics = self.quantitative_evaluation(val_data,config)
                val_metrics_results['ade'].append(val_metrics['ade'])
                val_metrics_results['fde'].append(val_metrics['fde'])
                if val_metrics["ade"]< best['ade']:
                    best['ade'] = val_metrics["ade"]
                    best['fde'] = val_metrics["fde"]
                    best["patchId"]= idx
                    # Save the best model so far
                    checkpoint.write(checkpoint_prefix+'-best')
                print('Epoch {}. Validation ADE {:.4f}'.format(epoch + 1, val_metrics['ade']))
        return train_loss_results,val_loss_results,val_metrics_results,best["patchId"]

    def quantitative_evaluation(self,test_data,config):
        l2dis = []
        num_batches_per_epoch = test_data.get_num_batches()
        for idx, batch in tqdm(test_data.get_batches(config.batch_size, num_steps = num_batches_per_epoch, shuffle=True), total = num_batches_per_epoch, ascii = True):
            # Format the data
            batch_inputs, batch_targets = get_batch(batch, config)
            pred_out               = self.batch_predict(batch_inputs,batch_targets.shape[1])
            this_actual_batch_size = batch["original_batch_size"]
            d = []
            # For all the trajectories in the batch
            for i, (obs_traj_gt, pred_traj_gt) in enumerate(zip(batch["obs_traj"], batch["pred_traj"])):
                if i >= this_actual_batch_size:
                    break
                # Conserve the x,y coordinates
                this_pred_out     = pred_out[i][:, :2] #[pred,2]
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
                d.append(diff)
            l2dis += d
        ade = [t for o in l2dis for t in o] # average displacement
        fde = [o[-1] for o in l2dis] # final displacement
        return { "ade": np.mean(ade), "fde": np.mean(fde)}

    # Perform a qualitative evaluation over a baych of n_trajectories
    def qualitative_evaluation(self,dataset,config,n_trajectories):
        traj_obs = []
        traj_gt  = []
        traj_pred= []
        # Select n_trajectories indices
        trajIds  = np.random.randint(dataset.get_data_size(),size=n_trajectories)
        # Form th batch
        batch    = dataset.get_by_idxs(trajIds)
        batch_inputs, batch_targets = get_batch(batch, config)
        # Perform prediction
        pred_traj                   = self.batch_predict(batch_inputs,batch_targets.shape[1])
        # Cycle over the instants to predict
        for i, (obs_traj_gt, pred_traj_gt) in enumerate(zip(batch["obs_traj"], batch["pred_traj"])):
            # Conserve the x,y coordinates
            this_pred_out     = pred_traj[i][:, :2]
            # Convert it to absolute (starting from the last observed position)
            if config.output_representation=='dxdy':
                this_pred_out_abs = relative_to_abs(this_pred_out, obs_traj_gt[-1])
            else:
                this_pred_out_abs = vw_to_abs(this_pred_out, obs_traj_gt[-1])

            # Keep all the trajectories
            traj_obs.append(obs_traj_gt)
            traj_gt.append(pred_traj_gt)
            traj_pred.append(this_pred_out_abs)
        # Plot ground truth and predictions
        plot_gt_preds(traj_gt,traj_obs,traj_pred)
