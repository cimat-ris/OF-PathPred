import tensorflow as tf
import os,logging,operator,functools,sys
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, losses
from .blocks import TrajectoryAndContextEncoder, TrajectoryDecoderInitializer
from .model_deterministic_rnn import PredictorDetRNN

""" Trajectory decoder.
    Generates samples for the next position
"""
class DecoderOf(tf.keras.Model):
    def __init__(self, config):
        super(DecoderOf, self).__init__(name="trajectory_decoder")
        self.rnn_type       = config.rnn_type
        # Linear embedding of the encoding resulting observed trajectories
        self.traj_xy_emb_dec = layers.Dense(config.emb_size,
            activation=config.activation_func,
            name='trajectory_position_embedding')
        # RNN cell
        # Condition for cell type
        if self.rnn_type == 'gru':
            # GRU cell
            self.dec_cell_traj = layers.GRUCell(config.dec_hidden_size,
                                                recurrent_initializer='glorot_uniform',
                                                dropout=config.dropout_rate,
                                                recurrent_dropout=config.dropout_rate,
                                                name='trajectory_decoder_cell_with_GRU')
        else:
            # LSTM cell
            self.dec_cell_traj = layers.LSTMCell(config.dec_hidden_size,
                                                recurrent_initializer='glorot_uniform',
                                                name='trajectory_decoder_cell_with_LSTM',
                                                dropout=config.dropout_rate,
                                                recurrent_dropout=config.dropout_rate)
        # RNN layer
        self.recurrentLayer = layers.RNN(self.dec_cell_traj,return_sequences=True,return_state=True)
        # Dropout layer
        self.dropout = layers.Dropout(config.dropout_rate,name="dropout_decoder_h")
        # Mapping from h to positions
        self.h_to_xy = layers.Dense(config.P,activation=tf.identity,name='h_to_xy')

        # Input layers
        # Position input
        dec_input_shape      = (1,config.P)
        self.input_layer_pos = layers.Input(dec_input_shape,name="position")
        enc_last_state_shape = (config.dec_hidden_size)
        # Proposals for inital states
        self.input_layer_hid1= layers.Input(enc_last_state_shape,name="initial_state_h")
        self.input_layer_hid2= layers.Input(enc_last_state_shape,name="initial_state_c")
        self.out = self.call((self.input_layer_pos,(self.input_layer_hid1,self.input_layer_hid2)))
        # Call init again. This is a workaround for being able to use summary
        super(DecoderOf, self).__init__(
                    inputs= [self.input_layer_pos,self.input_layer_hid1,self.input_layer_hid2],
                    outputs=self.out)

    # Call to the decoder
    def call(self, inputs, training=None):
        dec_input, last_states = inputs
        # Embedding from positions
        decoder_inputs_emb = self.traj_xy_emb_dec(dec_input)
        # Application of the RNN: outputs are [N,1,dec_hidden_size],[N,dec_hidden_size],[N,dec_hidden_size]
        if (self.rnn_type=='gru'):
            outputs    = self.recurrentLayer(decoder_inputs_emb,initial_state=last_states[0],training=training)
            # Last h state repeated to have always 2 tensors in cur_states
            cur_states = outputs[1:2]
            cur_states.append(outputs[1:2])
        else:
            outputs    = self.recurrentLayer(decoder_inputs_emb,initial_state=last_states,training=training)
            # Last h,c states
            cur_states = outputs[1:3]
        # Apply dropout layer on the h  state before mapping to positions x,y
        decoder_latent = self.dropout(cur_states[0],training=training)
        decoder_latent = tf.expand_dims(decoder_latent,1)
        # Something new: we try to learn the residual to the constant velocity case
        # Hence the output is equal to th input plus what we learn
        decoder_out_xy = self.h_to_xy(decoder_latent) + dec_input
        return decoder_out_xy, cur_states

# The main class
class PredictorMultOf():

    """
    Model parameters.
    """
    class Parameters(PredictorDetRNN.parameters):
        def __init__(self, rnn_type='lstm'):
            super(PredictorMultOf.Parameters, self).__init__(rnn_type)
            # -----------------
            self.stack_rnn_size = 2
            self.output_var_dirs= 0
            # Optical flow
            self.flow_size      = 64
            self.add_social     = True
            self.rnn_type       = rnn_type

    # Constructor
    def __init__(self,config):
        logging.info("Initialization")
        # Flags for considering social interations
        self.stack_rnn_size = config.stack_rnn_size
        self.output_samples = 2*config.output_var_dirs+1
        self.output_var_dirs= config.output_var_dirs

        #########################################################################################
        # The components of our model are instantiated here
        # Encoder: Positions and context
        self.enc = TrajectoryAndContextEncoder(config)
        self.enc.summary()
        # Encoder to decoder initialization
        self.enctodec = TrajectoryDecoderInitializer(config)
        self.enctodec.summary()
        # Decoder
        self.dec = DecoderOf(config)
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
        self.loss_fn       = losses.LogCosh()
        self.loss_fn_local = losses.LogCosh(losses.Reduction.NONE)

    # Single training/testing step, for one batch: batch_inputs are the observations, batch_targets are the targets
    def batch_step(self, batch_inputs, batch_targets, metrics, training=True):
        traj_obs_inputs = batch_inputs[0]
        # Last observed position from the trajectory
        traj_obs_last = traj_obs_inputs[:, -1]
        # Variables to be trained
        variables = self.enc.trainable_weights + self.dec.trainable_weights
        # Open a GradientTape to record the operations run during the forward pass, which enables auto-differentiation.
        # The total loss will be accumulated on this variable
        loss_value = 0
        with tf.GradientTape() as g:
            #########################################################################################
            # Encoding
            last_states, context, obs_classif_logits = self.enc(batch_inputs, training=training)
            #########################################################################################
            # Mapping encoding to state of the decoder
            traj_cur_states_set = self.enctodec(last_states)
            #########################################################################################
            # Decoding is done here
            # Iterate over these possible initializing states
            losses_over_samples = []
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
                    t_pred, dec_states = self.dec([dec_input,traj_cur_states],training=training)
                    t_target           = tf.expand_dims(batch_targets[:, t], 1)
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
                losses_over_samples.append(tf.squeeze(loss_values,axis=1))
            # Stack into a tensor batch_size x self.output_samples
            losses_over_samples  = tf.stack(losses_over_samples, axis=1)
            closest_samples      = tf.math.argmin(losses_over_samples, axis=1)
            #########################################################################################
            softmax_samples      = tf.nn.softmax(-losses_over_samples/0.01, axis=1)
            metrics['obs_classif_sca'].update_state(closest_samples,obs_classif_logits)
            loss_value  += 0.005* tf.reduce_sum(losses.kullback_leibler_divergence(softmax_samples,obs_classif_logits))/losses_over_samples.shape[0]
            #########################################################################################
            # Losses are accumulated here
            ortho_cost  = 0.005*self.enctodec.ortho_loss()
            loss_value +=   ortho_cost
            # Get the vector of losses at the minimal value for each sample of the batch
            losses_at_min= tf.gather_nd(losses_over_samples,tf.stack([range(losses_over_samples.shape[0]),closest_samples],axis=1))
            # Sum over the batches, divided by the batch size
            loss_value  += tf.reduce_sum(losses_at_min)/losses_over_samples.shape[0]
            # TODO: tune this value in a more principled way?
            # L2 weight decay
            loss_value  += tf.add_n([ tf.nn.l2_loss(v) for v in variables
                        if 'bias' not in v.name ]) * 0.0008
            #########################################################################################

        #########################################################################################
        # Gradients and parameters update
        if training==True:
            # Get the gradients
            grads = g.gradient(loss_value, variables)
            # Run one step of gradient descent
            self.optimizer.apply_gradients(zip(grads, variables))
        #########################################################################################

        # Average loss over the predicted times
        batch_loss = (loss_value / int(batch_targets.shape[1]))
        return batch_loss


    # Prediction (testing) for one batch
    def predict(self, batch_inputs, n_steps):
        traj_obs_inputs = batch_inputs[0]
        # Last observed position from the trajectories
        traj_obs_last     = traj_obs_inputs[:, -1]
        # Feed-forward start here
        last_states, context, obs_classif_logits = self.enc(batch_inputs, training=False)
        # Mapping encoding to state of the decoder
        traj_cur_states_set = self.enctodec(last_states)

        # This will store the trajectories and the attention weights
        traj_pred_set  = []

        # Iterate over these possible initializing states
        for k in range(self.output_samples):
            # List for the predictions and attention weights
            traj_pred   = []
            # Decoder state is initialized here
            traj_cur_states  = traj_cur_states_set[k]
            # The first input to the decoder is the last observed position [Nx1xK]
            dec_input = tf.expand_dims(traj_obs_last, 1)
            # Iterate over timesteps
            for t in range(0, n_steps):
                # ------------------------ xy decoder--------------------------------------
                # Passing enc_output to the decoder
                t_pred, dec_states = self.dec([dec_input,traj_cur_states],training=False)
                # Next input is the last predicted position
                dec_input = t_pred
                # Add it to the list of predictions
                traj_pred.append(t_pred)
                # Reuse the hidden states for the next step
                traj_cur_states = dec_states
            traj_pred   = tf.squeeze(tf.stack(traj_pred, axis=1))
            traj_pred_set.append(traj_pred)
        # Results as tensors
        traj_pred_set   = tf.stack(traj_pred_set,  axis=1)
        return traj_pred_set
