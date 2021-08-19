import tensorflow as tf
import os,logging,operator,functools,sys
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, losses
from .modules import TrajectoryEncoder, SocialEncoder, FocalAttention,TrajectoryDecoderInitializer, ObservedTrajectoryClassifier
from .model_deterministic_rnn import BasicRNNModelParameters
"""
Model parameters.
"""
class ModelParameters(BasicRNNModelParameters):
    def __init__(self, add_kp=False, add_social=False, rnn_type='lstm'):
        super(ModelParameters, self).__init__(rnn_type)
        # -----------------
        self.add_kp         = add_kp
        self.add_social     = add_social
        self.stack_rnn_size = 2
        self.output_var_dirs= 0
        # Key points
        self.kp_size        = 18
        # Optical flow
        self.flow_size      = 64
        self.mc_samples     = 20
        self.rnn_type       = rnn_type

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
        # Encoder to decoder initialization
        self.enctodec         = TrajectoryDecoderInitializer(config)
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
            return self.enctodec([traj_last_states[0],soc_last_states]), context, obs_classif_logits
        else:
            #return traj_last_states, obs_enc_h
            return self.enctodec([traj_last_states[0]]), context, obs_classif_logits

""" Trajectory decoder.
    Generates samples for the next position
"""
class TrajectoryDecoder(tf.keras.Model):
    def __init__(self, config):
        super(TrajectoryDecoder, self).__init__(name="trajectory_decoder")
        self.add_social     = config.add_social
        self.stack_rnn_size = config.stack_rnn_size
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
                                                name='trajectory_decoder_cell',
                                                dropout=config.dropout_rate,
                                                recurrent_dropout=config.dropout_rate)
        # RNN layer
        self.recurrentLayer = layers.RNN(self.dec_cell_traj,return_sequences=True,return_state=True)
        self.M = 1
        if (self.add_social):
            self.M=self.M+1

        # Attention layer
        self.focal_attention = FocalAttention(config,self.M)
        # Dropout layer
        self.dropout = layers.Dropout(config.dropout_rate,name="dropout_decoder_h")
        # Mapping from h to positions
        self.h_to_xy = layers.Dense(config.P,
            activation=tf.identity,
            name='h_to_xy')

        # Input layers
        # Position input
        dec_input_shape      = (1,config.P)
        self.input_layer_pos = layers.Input(dec_input_shape,name="position")
        enc_last_state_shape = (config.dec_hidden_size)
        # Proposals for inital states
        self.input_layer_hid1= layers.Input(enc_last_state_shape,name="initial_state_h")
        self.input_layer_hid2= layers.Input(enc_last_state_shape,name="initial_state_c")
        # Context shape: [N,M,T1,h_dim]
        ctxt_shape = (self.M,config.obs_len,config.enc_hidden_size)
        # Context input
        self.input_layer_ctxt = layers.Input(ctxt_shape,name="context")
        self.out = self.call((self.input_layer_pos,(self.input_layer_hid1,self.input_layer_hid2),self.input_layer_ctxt))
        # Call init again. This is a workaround for being able to use summary
        super(TrajectoryDecoder, self).__init__(
                    inputs= [self.input_layer_pos,self.input_layer_hid1,self.input_layer_hid2,self.input_layer_ctxt],
                    outputs=self.out)

    # Call to the decoder
    def call(self, inputs, training=None):
        dec_input, last_states, context = inputs
        # Embedding from positions
        decoder_inputs_emb = self.traj_xy_emb_dec(dec_input)
        # context: [N,1,h_dim]
        # query is the last h so far: [N,h_dim]. Since last_states is a pair (h,c), we take last_states[0]
        query              = last_states[0]
        # Use attention to define the augmented input here
        attention, Wft  = self.focal_attention(query, context)
        # TODO: apply the embedding to the concatenation instead of just on the first part (positions)
        # Augmented input: [N,1,h_dim+emb]
        augmented_inputs= tf.concat([decoder_inputs_emb, attention], axis=2)
        # Application of the RNN: outputs are [N,1,dec_hidden_size],[N,dec_hidden_size],[N,dec_hidden_size]
        if (self.rnn_type=='gru'):
            outputs    = self.recurrentLayer(augmented_inputs,initial_state=last_states[0],training=training)
            # Last h state repeated to have always 2 tensors in cur_states
            cur_states = outputs[1:2]
            cur_states.append(outputs[1:2])
        else:
            outputs    = self.recurrentLayer(augmented_inputs,initial_state=last_states,training=training)
            # Last h,c states
            cur_states = outputs[1:3]
        # Apply dropout layer on the h  state before mapping to positions x,y
        decoder_latent = self.dropout(cur_states[0],training=training)
        decoder_latent = tf.expand_dims(decoder_latent,1)
        # Something new: we try to learn the residual to the constant velocity case
        # Hence the output is equal to th input plus what we learn
        decoder_out_xy = self.h_to_xy(decoder_latent) + dec_input
        return decoder_out_xy, cur_states, Wft

# The main class
class TrajectoryEncoderDecoder():
    # Constructor
    def __init__(self,config):
        logging.info("Initialization")
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
        self.loss_fn       = losses.LogCosh()
        self.loss_fn_local = losses.LogCosh(losses.Reduction.NONE)

    # Trick to reset the weights: We save them and reload them
    def save_tmp(self):
        self.enc.save_weights('tmp_enc.h5')
        self.dec.save_weights('tmp_dec.h5')
    def load_tmp(self):
        self.enc.load_weights('tmp_enc.h5')
        self.dec.load_weights('tmp_dec.h5')

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
            # Encoding is done here
            traj_cur_states_set, context, obs_classif_logits = self.enc(batch_inputs, training=training)

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
                    t_pred, dec_states, __ = self.dec([dec_input,traj_cur_states,context],training=training)
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
        traj_cur_states_set, context, obs_classif_logits = self.enc(batch_inputs, training=False)
        # This will store the trajectories and the attention weights
        traj_pred_set  = []
        att_weights_set= []

        # Iterate over these possible initializing states
        for k in range(self.output_samples):
            # List for the predictions and attention weights
            traj_pred   = []
            att_weights = []
            # Decoder state is initialized here
            traj_cur_states  = traj_cur_states_set[k]
            # The first input to the decoder is the last observed position [Nx1xK]
            dec_input = tf.expand_dims(traj_obs_last, 1)
            # Iterate over timesteps
            for t in range(0, n_steps):
                # ------------------------ xy decoder--------------------------------------
                # Passing enc_output to the decoder
                t_pred, dec_states, wft = self.dec([dec_input,traj_cur_states,context],training=False)
                # Next input is the last predicted position
                dec_input = t_pred
                # Add it to the list of predictions
                traj_pred.append(t_pred)
                att_weights.append(wft)
                # Reuse the hidden states for the next step
                traj_cur_states = dec_states
            traj_pred   = tf.squeeze(tf.stack(traj_pred, axis=1))
            att_weights = tf.squeeze(tf.stack(att_weights, axis=1))
            traj_pred_set.append(traj_pred)
            att_weights_set.append(att_weights)
        # Results as tensors
        traj_pred_set   = tf.stack(traj_pred_set,  axis=1)
        att_weights_set = tf.stack(att_weights_set,axis=1)
        return traj_pred_set,att_weights_set
