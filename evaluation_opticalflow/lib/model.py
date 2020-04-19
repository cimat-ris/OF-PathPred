import tensorflow as tf
import functools
import operator
import numpy as np

class Model(object):
    """Model graph definitions.
    """
    def __init__(self, config):
        self.scope       = config.modelname
        self.config      = config
        self.global_step = tf.Variable(initial_value=0,name='global_step', shape=[],dtype='int32',trainable=False)

        # Get all the dimension here
        # Tensor dimensions, so pylint: disable=g-bad-name
        N  = self.N  = config.batch_size
        KP = self.KP = config.kp_size ####
        P  = self.P  = 2              # Spatial coordinates
        T1 = config.obs_len           # Length of the observations

        # The trajactory sequence: [N,T1,2] # T1 is the obs_len
        # in training, it is the obs+pred combined,
        # in testing, only obs is fed and the rest is zeros
        # mask is used for variable length input extension
        self.traj_obs_gt      = tf.compat.v1.placeholder('float', [N, None, P], name='traj_obs_gt')
        self.traj_obs_gt_mask = tf.compat.v1.placeholder('bool', [N, None], name='traj_obs_gt_mask')

        #[N,T2,2]
        self.traj_pred_gt      = tf.compat.v1.placeholder('float', [N, None, P], name = 'traj_pred_gt')
        self.traj_pred_gt_mask = tf.compat.v1.placeholder('bool', [N, None], name = 'traj_pred_gt_mask')

        # Info about keypoints
        self.obs_kp = tf.compat.v1.placeholder('float', [N, None, KP, 2], name = 'obs_kp')
        # Info about optical flow
        self.obs_flow = tf.compat.v1.placeholder('float',[N, None,64],name='obsf_flow')
        # Flag for trainig. Used for drop out switch
        self.is_train  = tf.compat.v1.placeholder('bool', [], name = 'is_train')
        # Loss function
        self.loss = None
        # Build foward model
        self.build_forward()
        # Build loss
        self.build_loss()


    def build_forward(self):
        """Build the forward model graph."""
        config = self.config
        # Tensor dimensions, so pylint: disable=g-bad-name
        N  = self.N  # Batches
        KP = self.KP # Number of keypoints

        # Add dropout
        keep_prob = tf.cond(self.is_train,
                        lambda: tf.constant(config.keep_prob),
                        lambda: tf.constant(1.0))
        # ------------------------- Encoder ------------------------
        # Trajectory encoder
        enc_cell_traj = tf.compat.v1.nn.rnn_cell.LSTMCell(
            config.enc_hidden_size, state_is_tuple=True, name='enc_traj')
        enc_cell_traj = tf.compat.v1.nn.rnn_cell.DropoutWrapper(enc_cell_traj, keep_prob)

        # Person pose (keypoints) encoder
        if config.add_kp:
            enc_cell_kp = tf.compat.v1.nn.rnn_cell.LSTMCell(config.enc_hidden_size, state_is_tuple=True, name='enc_kp')
            enc_cell_kp = tf.compat.v1.nn.rnn_cell.DropoutWrapper(enc_cell_kp, keep_prob)

        # Social encoding part (optical flow)
        if config.add_social:
            enc_cell_soc = tf.compat.v1.nn.rnn_cell.LSTMCell(
                config.enc_hidden_size,state_is_tuple = True,name='enc_social')
            enc_cell_soc = tf.compat.v1.nn.rnn_cell.DropoutWrapper(enc_cell_soc,keep_prob)

        # ------------------------ Decoder
        if config.multi_decoder:
            dec_cell_traj = [tf.compat.v1.nn.rnn_cell.LSTMCell(
                config.dec_hidden_size, state_is_tuple=True, name='dec_traj_%s' % i)
                             for i in range(len(config.traj_cats))]
            dec_cell_traj = [tf.compat.v1.nn.rnn_cell.DropoutWrapper(one, keep_prob) for one in dec_cell_traj]
        else:
            dec_cell_traj = tf.compat.v1.nn.rnn_cell.LSTMCell(config.dec_hidden_size, state_is_tuple=True, name='dec_traj')
            dec_cell_traj = tf.compat.v1.nn.rnn_cell.DropoutWrapper(dec_cell_traj, keep_prob)

        # ----------------------------------------------------------
        # the obs part is the same for training and testing
        # obs_out is only used in training
        # encoder, decoder
        # top_scope is used for variable inside
        # encode and decode if want to share variable across
        with tf.compat.v1.variable_scope('person_pred') as top_scope:
            # xy encoder: [N,T1,h_dim]
            obs_length = tf.reduce_sum(tf.cast(self.traj_obs_gt_mask, 'int32'), 1)
            # Linear embedding
            traj_xy_emb_enc = linear(self.traj_obs_gt,
                               output_size=config.emb_size,
                               activation=config.activation_func,
                               add_bias=True,
                               scope='enc_xy_emb')
            # Applies the position sequence through the LSTM
            traj_obs_enc_h, traj_obs_enc_last_state = tf.compat.v1.nn.dynamic_rnn(
                enc_cell_traj, traj_xy_emb_enc, sequence_length = obs_length,
                dtype='float', scope='encoder_traj')
            # Get the hidden states and the last hidden state, separately
            enc_h_list          = [traj_obs_enc_h]
            enc_last_state_list = [traj_obs_enc_last_state]

            # Person pose (keypoints)
            if config.add_kp:
                obs_kp = tf.reshape(self.obs_kp, [N, -1, KP*2])
                # Linear embedding of the keypoints
                obs_kp = linear(obs_kp, output_size=config.emb_size, add_bias=True,
                                activation=config.activation_func, scope='kp_emb')
                # Applies the person pose (keypoints) sequence through the LSTM
                kp_obs_enc_h, kp_obs_enc_last_state = tf.nn.dynamic_rnn(
                    enc_cell_kp, obs_kp, sequence_length=obs_length, dtype='float',
                    scope='encoder_kp')
                # Get the hidden states and the last hidden state, separately
                enc_h_list.append(kp_obs_enc_h)
                enc_last_state_list.append(kp_obs_enc_last_state)

            # Interaccion social through optical flow
            if config.add_social:
                # Linear embedding of the optical flow
                obs_soc = linear(self.obs_flujo,output_size = config.emb_size,add_bias = True,
                    activation=config.activation_func,scope='flujo_emb')
                # Applies the person pose (keypoints) sequence through the LSTM
                soc_obs_enc_h, soc_obs_enc_last_state =tf.nn.dynamic_rnn(
                    enc_cell_soc, obs_soc, sequence_length=obs_length, dtype='float',
                    scope='encoder_soc')
                # Get the hidden states and the last hidden state, separately
                enc_h_list.append(soc_obs_enc_h)
                enc_last_state_list.append(soc_obs_enc_last_state)

            # Pack all observed hidden states
            # [batch,m,obs,h_dim]
            obs_enc_h          = tf.stack(enc_h_list, axis=1)
            obs_enc_last_state = concat_states(enc_last_state_list, axis=1)

            # -------------------------------------------------- xy decoder
            # Last observed position
            traj_obs_last = self.traj_obs_gt[:, -1]
            pred_length = tf.reduce_sum(
                tf.cast(self.traj_pred_gt_mask, 'int32'), 1)  # N

            # Multiple decoder
            if config.multi_decoder:
                # [N, num_traj_cat] # each is num_traj_cat classification
                self.traj_class_logits = self.traj_class_head(
                    obs_enc_h, obs_enc_last_state, scope='traj_class_predict')

                # [N]
                traj_class = tf.argmax(self.traj_class_logits, axis=1)

                traj_class_gated = tf.cond(
                    self.is_train,
                    lambda: self.traj_class_gt,
                    lambda: traj_class,
                )

                traj_pred_outs = [
                    self.decoder(
                        traj_obs_last,
                        traj_obs_enc_last_state,
                        obs_enc_h,
                        pred_length,
                        dec_cell_traj[traj_cat],
                        top_scope=top_scope,
                        scope='decoder_%s' % traj_cat)
                    for _, traj_cat in config.traj_cats
                ]

                # [N, num_decoder, T, 2]
                self.traj_pred_outs = tf.stack(traj_pred_outs, axis=1)

                # [N, 2]
                indices = tf.stack(
                    [tf.range(N), tf.to_int32(traj_class_gated)], axis=1)

                # [N, T, 2]
                traj_pred_out = tf.gather_nd(self.traj_pred_outs, indices)

            else:
                # Single decoder called: takes the last observed position, the last encoding state,
                # the set of all hidden states, the number of prediction steps, and the decoder cell.
                traj_pred_out = self.decoder(traj_obs_last, traj_obs_enc_last_state,
                                             obs_enc_h, pred_length, dec_cell_traj,
                                             top_scope=top_scope, scope='decoder')
        # For loss and forward
        self.traj_pred_out = traj_pred_out

    # Decoder
    def decoder(self, first_input, enc_last_state, enc_h, pred_length, rnn_cell,top_scope, scope):
        """Decoder definition."""
        config = self.config
        # Tensor dimensions, so pylint: disable=g-bad-name
        N = self.N # Batches
        P = self.P # Spatial dimension

        with tf.compat.v1.variable_scope(scope):
            # This is only used for training
            with tf.compat.v1.name_scope('prepare_pred_gt_training'):
                # These input only used during training
                time_1st_traj_pred = tf.transpose(
                    self.traj_pred_gt, perm=[1, 0, 2])  # [N,T2,W] -> [T2,N,W]
                T2 = tf.shape(time_1st_traj_pred)[0]  # T2
                traj_pred_gt = tf.TensorArray(size= T2, dtype='float')
                traj_pred_gt = traj_pred_gt.unstack(
                    time_1st_traj_pred)  # [T2] , [N,W]

            # all None for first call
            with tf.compat.v1.name_scope('decoder_rnn'):
                def decoder_loop_fn(time, cell_output, cell_state, loop_state):
                    """RNN loop function for the decoder."""
                    emit_output = cell_output  # == None for time==0

                    elements_finished = time >= pred_length
                    finished          = tf.reduce_all(elements_finished)

                    # h_{t-1}
                    with tf.compat.v1.name_scope('prepare_next_cell_state'):
                        if cell_output is None:
                            next_cell_state = enc_last_state
                        else:
                            next_cell_state = cell_state
                    # x_t
                    with tf.compat.v1.name_scope('prepare_next_input'):
                        if cell_output is None:  # first time
                            next_input_xy = first_input  # the last observed x,y as input
                        else:
                            # for testing, construct from this output to be next input
                            next_input_xy = tf.cond(
                                # first check the sequence finished or not
                                finished,
                                lambda: tf.zeros([N, P], dtype='float'),
                                # pylint: disable=g-long-lambda
                                lambda: tf.cond(
                                    self.is_train,
                                    # this will make training faster than testing
                                    lambda: traj_pred_gt.read(time),
                                    # hidden vector from last step to coordinates
                                    lambda: self.hidden2xy(cell_output, scope=top_scope,
                                                           additional_scope='hidden2xy'))
                            )
                        # spatial embedding
                        # [N,emb]
                        xy_emb = linear(next_input_xy, output_size=config.emb_size,
                            activation=config.activation_func, add_bias=True,
                            scope='xy_emb_dec')

                        next_input = xy_emb
                        with tf.compat.v1.name_scope('attend_enc'):
                            # [N,h_dim]
                            attended_encode_states = focal_attention(
                                next_cell_state.h, enc_h, use_sigmoid=False,
                                scope='decoder_attend_encoders')

                            next_input = tf.concat(
                                [xy_emb, attended_encode_states], axis=1)
                    return elements_finished, next_input, next_cell_state,emit_output, None  # next_loop_state

                decoder_out_ta, _, _ = tf.compat.v1.nn.raw_rnn(
                    rnn_cell, decoder_loop_fn, scope='decoder_rnn')
            with tf.compat.v1.name_scope('reconstruct_output'):

                decoder_out_h = decoder_out_ta.stack()  # [T2,N,h_dim]
                # [N,T2,h_dim]
                decoder_out_h = tf.transpose(decoder_out_h, perm=[1, 0, 2])
            # recompute the output;
            # if use loop_state to save the output, will 10x slower

            # use the same hidden2xy for different decoder
            decoder_out = self.hidden2xy(
                decoder_out_h, scope=top_scope, additional_scope='hidden2xy')
        return decoder_out

    def hidden2xy(self, lstm_h, return_scope=False, scope='hidden2xy',additional_scope=None):
        """Hiddent states to xy coordinates."""
        # Tensor dimensions, so pylint: disable=g-bad-name
        P = self.P

        with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE) as this_scope:
            if additional_scope is not None:
                return self.hidden2xy(lstm_h, return_scope=return_scope,
                                      scope=additional_scope, additional_scope=None)
            # Dense layer
            out_xy = linear(lstm_h, output_size=P, activation=tf.identity,
                            add_bias=False, scope='out_xy_mlp2')

            if return_scope:
                return out_xy, this_scope
            return out_xy

    def build_loss(self):
        """Model loss."""
        config = self.config
        losses = []
        # N,T,W
        # L2 loss
        # [N,T2,W]
        traj_pred_out = self.traj_pred_out
        traj_pred_gt = self.traj_pred_gt
        diff = traj_pred_out - traj_pred_gt

        xyloss = tf.pow(diff, 2)  # [N,T2,2]
        xyloss = tf.reduce_mean(xyloss)
        self.xyloss = xyloss

        losses.append(xyloss)
        #self.loss = tf.add_n(losses, name='total_losses')
        self.loss = self.xyloss

    def get_feed_dict(self, data, is_train):
        """Given a batch of data, construct the feed dict."""
        # get the cap for each kind of step first
        config = self.config
        # Tensor dimensions, so pylint: disable=g-bad-name
        N = self.N
        P = self.P
        KP = self.KP ####
        #MNP = self.MNP

        T_in = config.obs_len
        T_pred = config.pred_len

        feed_dict = {}

        #initial all the placeholder
        traj_obs_gt = np.zeros([N, T_in, P], dtype='float')
        traj_obs_gt_mask = np.zeros([N, T_in], dtype='bool')

        #link the feed_dict
        feed_dict[self.traj_obs_gt] = traj_obs_gt
        feed_dict[self.traj_obs_gt_mask] = traj_obs_gt_mask

        #for getting pred length during test time
        traj_pred_gt_mask = np.zeros([N, T_pred], dtype='bool')
        feed_dict[self.traj_pred_gt_mask] = traj_pred_gt_mask

        #this is needed since it is in tf.conf?
        traj_pred_gt = np.zeros([N, T_pred, P], dtype='float')
        feed_dict[self.traj_pred_gt] = traj_pred_gt  # all zero when testing,

        feed_dict[self.is_train] = is_train
        #encoder features
        # ------------------------------------- xy input

        assert len(data['obs_traj_rel']) == N
        for i, (obs_data, pred_data) in enumerate(zip(data['obs_traj_rel'],
                                                  data['pred_traj_rel'])):
            for j, xy in enumerate(obs_data):
                traj_obs_gt[i, j, :] = xy
                traj_obs_gt_mask[i, j] = True
            for j in range(config.pred_len):
                # used in testing to get the preiction length
                traj_pred_gt_mask[i, j] = True

        # ------------------------------------------------------
        if config.add_social:
            #assert len(data['obs_person_rel'])== N
            obs_flujo = np.zeros((N, T_in, 64),dtype ='float')
            feed_dict[self.obs_flujo] = obs_flujo

            #each batch
            # agregamos el vector de flujo
            #for i, flujo_seq in enumerate(obs_flujo):
            #    for j, flujo_paso in enumerate(flujo_seq):
            #        obs_flujo[i,j,:]= flujo_paso
            #feed_dict[self.obs_frame] = data['obs_person']
            for i, flujo_seq in enumerate(data['obs_flujo']):
                for j , flujo_paso in enumerate(flujo_seq):
                    obs_flujo[i,j,:] = flujo_paso

        # -----------------------------------------------------------
        # person pose input
        if config.add_kp:
            obs_kp = np.zeros((N, T_in, KP, 2), dtype='float')
            feed_dict[self.obs_kp] = obs_kp
            # each bacth
            for i, obs_kp_rel in enumerate(data['obs_kp_rel']):
                for j, obs_kp_step in enumerate(obs_kp_rel):
                    obs_kp[i, j, :, :] = obs_kp_step
        # ----------------------------training
        if is_train:
            for i, (obs_data, pred_data) in enumerate(zip(data['obs_traj_rel'],
                                                    data['pred_traj_rel'])):
                for j, xy in enumerate(pred_data):
                    traj_pred_gt[i, j, :] = xy
                    traj_pred_gt_mask[i, j] = True
        return feed_dict

def reconstruct(tensor, ref, keep):
    """Reverse the flatten function.
    Args:
    tensor: the tensor to operate on
    ref: reference tensor to get original shape
    keep: index of dim to keep
    Returns:
    Reconstructed tensor
    """
    ref_shape = ref.get_shape().as_list()
    tensor_shape = tensor.get_shape().as_list()
    ref_stop = len(ref_shape) - keep
    tensor_start = len(tensor_shape) - keep
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i]
                  for i in range(tensor_start, len(tensor_shape))]
    # keep_shape = tensor.get_shape().as_list()[-keep:]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    return out

def flatten(tensor, keep):
    """Flatten a tensor.
    keep how many dimension in the end, so final rank is keep + 1
    [N,M,JI,JXP,dim] -> [N*M*JI,JXP,dim]
    Args:
    tensor: the tensor to operate on
    keep: index of dim to keep
    Returns:
    Flattened tensor
    """
    # get the shape
    fixed_shape = tensor.get_shape().as_list()  # [N, JQ, di] # [N, M, JX, di]
    # len([N, JQ, di]) - 2 = 1 # len([N, M, JX, di] ) - 2 = 2
    start = len(fixed_shape) - keep
    # each num in the [] will a*b*c*d...
    # so [0] -> just N here for left
    # for [N, M, JX, di] , left is N*M
    left = functools.reduce(operator.mul, [fixed_shape[i] or tf.shape(tensor)[i]
                               for i in range(start)])
    # [N, JQ,di]
    # [N*M, JX, di]
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i]
                        for i in range(start, len(fixed_shape))]
    # reshape
    flat = tf.reshape(tensor, out_shape)
    return flat

def softmax(logits, scope=None):
    """a flatten and reconstruct version of softmax."""
    with tf.compat.v1.name_scope(scope or 'softmax'):
        flat_logits = flatten(logits, 1)
        flat_out = tf.nn.softmax(flat_logits)
        out = reconstruct(flat_out, logits, 1)
        return out

def softsel(target, logits, use_sigmoid=False, scope=None):
    """Apply attention weights."""

    with tf.compat.v1.variable_scope(scope or 'softsel'):  # no new variable tho
        if use_sigmoid:
            a = tf.nn.sigmoid(logits)
        else:
            a = softmax(logits)  # shape is the same
        target_rank = len(target.get_shape().as_list())
        # [N,M,JX,JQ,2d] elem* [N,M,JX,JQ,1]
        # second last dim
        return tf.reduce_sum(tf.expand_dims(a, -1)*target, target_rank-2)


def linear(x, output_size, scope, add_bias=False, wd=None, return_scope=False,
           reuse=None, activation=tf.identity, keep=1, additional_scope=None):

    """Fully-connected layer."""
    with tf.compat.v1.variable_scope(scope or 'xy_emb', reuse=tf.compat.v1.AUTO_REUSE) as this_scope:
        if additional_scope is not None:
            return linear(x, output_size, scope=additional_scope, add_bias=add_bias,
                          wd=wd, return_scope=return_scope, reuse=reuse,
                          activation=activation, keep=keep, additional_scope=None)

        # since the input here is not two rank,
        # we flat the input while keeping the last dims
        # keeping the last one dim # [N,M,JX,JQ,2d] => [N*M*JX*JQ,2d]
        flat_x = flatten(x, keep)
        # print flat_x.get_shape() # (?, 200) # wd+cwd
        bias_start = 0.0
        # need to be get_shape()[k].value
        if not isinstance(output_size, int):
            output_size = output_size.value

        def init(shape, dtype, partition_info):
            dtype = dtype
            partition_info = partition_info
            return tf.random.truncated_normal(shape, stddev=0.1)
        # Common weight tensor name, so pylint: disable=g-bad-name
        W = tf.compat.v1.get_variable('W', dtype='float', initializer=init,
                                shape=[flat_x.get_shape()[-1].value, output_size])
        #W = tf.Variable('W', dtype='float', initializer=init,shape=[flat_x.get_shape()[-1].value, output_size])
        flat_out = tf.matmul(flat_x, W)
        if add_bias:
            # disable=unused-argument
            def init_b(shape, dtype, partition_info):
                dtype = dtype
                partition_info = partition_info
                return tf.constant(bias_start, shape=shape)

            bias = tf.compat.v1.get_variable('b', dtype='float', initializer=init_b,
                                   shape=[output_size])
            flat_out += bias

        flat_out = activation(flat_out)
        out = reconstruct(flat_out, x, keep)
        if return_scope:
            return out, this_scope
        else:
            return out

def focal_attention(query, context, use_sigmoid=False, scope=None):
    """Focal attention layer.
    Args:
    query : [N, dim1]
    context: [N, num_channel, T, dim2]
    use_sigmoid: use sigmoid instead of softmax
    scope: variable scope
    Returns:
    Tensor
    """
    #print("*** focal attention ***")
    with tf.compat.v1.variable_scope(scope or 'attention', reuse=tf.compat.v1.AUTO_REUSE):
        # Tensor dimensions, so pylint: disable=g-bad-name
        _, d = query.get_shape().as_list()
        _, K, _, d2 = context.get_shape().as_list()
        assert d == d2

        T = tf.shape(context)[2]

        # [N,d] -> [N,K,T,d]
        query_aug = tf.tile(tf.expand_dims(tf.expand_dims(query, 1), 1), [1, K, T, 1])
        # cosine simi
        query_aug_norm = tf.nn.l2_normalize(query_aug, -1)
        context_norm = tf.nn.l2_normalize(context, -1)
        # [N, K, T]

        a_logits = tf.reduce_sum(tf.multiply(query_aug_norm, context_norm), 3)
        a_logits_maxed = tf.reduce_max(a_logits, 2)  # [N,K]

        attended_context = softsel(softsel(context, a_logits,
                                           use_sigmoid=use_sigmoid), a_logits_maxed,
                                   use_sigmoid=use_sigmoid)
        return attended_context

def concat_states(state_tuples, axis):
    """Concat LSTM states."""
    return tf.compat.v1.nn.rnn_cell.LSTMStateTuple(c=tf.concat([s.c for s in state_tuples],
                                                   axis=axis),
                                       h=tf.concat([s.h for s in state_tuples],
                                                   axis=axis))
