import math
import random
import itertools
import collections
import numpy as np

def grouper(lst, num):
    args = [iter(lst)]*num
    out = itertools.zip_longest(*args, fillvalue=None)
    out = list(out)
    return out

def get_batch(batch_data, config):
    """Given a batch of data, determine the input and ground truth."""
    N      = len(batch_data['obs_traj_rel'])
    P      = config.P
    OF     = config.flow_size
    T_in   = config.obs_len
    T_pred = config.pred_len

    returned_inputs = []
    traj_obs_gt  = np.zeros([N, T_in, P], dtype='float32')
    traj_pred_gt = np.zeros([N, T_pred, P], dtype='float32')
    # --- xy input
    for i, (obs_data, pred_data) in enumerate(zip(batch_data['obs_traj_rel'],
                                                  batch_data['pred_traj_rel'])):
        for j, xy in enumerate(obs_data):
            traj_obs_gt[i, j, :] = xy
        for j, xy in enumerate(pred_data):
            traj_pred_gt[i, j, :]   = xy
    returned_inputs.append(traj_obs_gt)
    # ------------------------------------------------------
    # Social component (through optical flow)
    if config.add_social:
        obs_flow = np.zeros((N, T_in, OF),dtype ='float32')
        # each batch
        for i, flow_seq in enumerate(batch_data['obs_flow']):
            for j , flow_step in enumerate(flow_seq):
                obs_flow[i,j,:] = flow_step
        returned_inputs.append(obs_flow)
    # -----------------------------------------------------------
    # Person pose input
    if config.add_kp:
        obs_kp = np.zeros((N, T_in, KP, 2), dtype='float32')
        # each bacth
        for i, obs_kp_rel in enumerate(batch_data['obs_kp_rel']):
            for j, obs_kp_step in enumerate(obs_kp_rel):
                obs_kp[i, j, :, :] = obs_kp_step
    return returned_inputs,traj_pred_gt

class Dataset(object):
    """Class for batching during training and testing."""
    def __init__(self, data, config):
        self.data         = data
        self.valid_idxs   = range(self.get_data_size())
        self.num_examples = len(self.valid_idxs)#4441
        self.config       = config

    def get_data_size(self):
        #num of examples
        return len(self.data["obs_traj"])

    def get_by_idxs(self, idxs):
        out = {}
        for key, val in self.data.items():
            if not key in out:
                out[key] = []
            out[key].extend(val[idx] for idx in idxs)
        return out

    def get_num_batches(self):
        """ Number of batches."""
        return int(math.ceil(self.num_examples / float(self.config.batch_size)))

    def get_batches(self, batch_size, num_steps = 0, shuffle = True, full = False):
        """Iterator to get batches.
        should return num_steps -> batches
        step is total/batchSize * epoch
        cap means limits max number of generated batches to 1 epoch
        Args:
        batch_size: batch size.
        num_steps: total steps.
        shuffle: whether shuffling the data
        Yields:
        Dataset object.
        """
        num_batches_per_epoch = int(math.ceil(self.num_examples / float(batch_size)))
        if full:
            num_steps = num_batches_per_epoch

        # this may be zero
        num_epochs = int(math.ceil(num_steps/float(num_batches_per_epoch)))

        # shuflle
        if shuffle:
            # All epoch has the same order.
            random_idxs = random.sample(self.valid_idxs, len(self.valid_idxs))
            # all batch idxs for one epoch

            def random_grouped():
                return list(grouper(random_idxs, batch_size))
            # grouper
            # given a list and n(batch_size), devide list into n sized chunks
            # last one will fill None
            grouped = random_grouped
        else:
            def raw_grouped():
                return list(grouper(self.valid_idxs, batch_size))
            grouped = raw_grouped

        # all batches idxs from multiple epochs
        batch_idxs_iter = itertools.chain.from_iterable(grouped() for _ in range(num_epochs))
        for _ in range(num_steps):  # num_step should be batch_idxs length
            # so in the end batch, the None will not included
            batch_idxs = tuple(i for i in next(batch_idxs_iter) if i is not None)  # each batch idxs
            # so batch_idxs might not be size batch_size
            # pad with the last item
            original_batch_size = len(batch_idxs)
            if len(batch_idxs) < batch_size:
                pad = batch_idxs[-1]
                batch_idxs = tuple(list(batch_idxs) + [pad for i in range(batch_size - len(batch_idxs))])

            # get the actual data based on idx
            batch_data = self.get_by_idxs(batch_idxs)
            batch_data.update({"original_batch_size": original_batch_size})
            yield batch_idxs, batch_data
