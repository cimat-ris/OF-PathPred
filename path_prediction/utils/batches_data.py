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

def get_batch(batch_data, config, rot='_rot'):
    """Given a batch of data, determine the input and ground truth."""
    N      = len(batch_data['obs_traj_rel'+rot])
    P      = config.P
    if hasattr(config, 'flow_size'):
        OF     = config.flow_size

    returned_inputs = []
    traj_obs_gt  = np.zeros([N, config.obs_len, P], dtype='float32')
    traj_pred_gt = np.zeros([N, config.pred_len, P], dtype='float32')
    # --- xy input
    for i, (obs_data, pred_data) in enumerate(zip(batch_data['obs_traj_rel'+rot],
                                                  batch_data['pred_traj_rel'+rot])):
        for j, xy in enumerate(obs_data):
            traj_obs_gt[i, j, :] = xy
        for j, xy in enumerate(pred_data):
            traj_pred_gt[i, j, :]   = xy
    returned_inputs.append(traj_obs_gt)
    # ------------------------------------------------------
    # Social component (through optical flow)
    if hasattr(config, 'add_social') and config.add_social:
        obs_flow = np.zeros((N, T_in, OF),dtype ='float32')
        # each batch
        for i, flow_seq in enumerate(batch_data['obs_optical_flow']):
            for j , flow_step in enumerate(flow_seq):
                obs_flow[i,j,:] = flow_step
        returned_inputs.append(obs_flow)
    # -----------------------------------------------------------
    return returned_inputs,traj_pred_gt
