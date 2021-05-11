import sys,os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import argparse
import time
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
print('[INF] Tensorflow version: ',tf.__version__)
tf.test.gpu_device_name()
import random
from datetime import datetime
random.seed(datetime.now())
import matplotlib.pyplot as plt

# Important imports
import path_prediction.batches_data
from path_prediction.training_utils import Experiment_Parameters
from path_prediction.interaction_optical_flow import OpticalFlowSimulator

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../trajnet++/trajnetplusplustools')))
import trajnetplusplustools
import logging
import socket


def prepare_data_trajnetplusplus(path, subset='/train/'):
    """ Prepares the train/val scenes and corresponding goals

    Parameters
    ----------
    subset: String ['/train/', '/val/']
        Determines the subset of data to be processed
    sample: Float (0.0, 1.0]
        Determines the ratio of data to be sampled

    Returns
    -------
    all_scenes: List
        List of all processed scenes
    all_goals: Dictionary
        Dictionary of goals corresponding to each dataset file.
        None if 'goals' argument is False.
    """

    ## read goal files
    all_scenes       = []
    seq_pos_all      = []
    seq_rel_all      = []
    seq_theta_all    = []
    seq_neighbors_all= []
    all_flow         = []
    all_vis_neigh    = []
    ## List file names
    files = [f.split('.')[-2] for f in os.listdir(path + subset) if f.endswith('.ndjson')]

    neighbors_n_max = 0
    ## Iterate over file names
    for file in files:
        reader = trajnetplusplustools.Reader(path + subset + file + '.ndjson', scene_type='paths')
        ## Necessary modification of train scene to add filename
        scene = [(file, s_id, s) for s_id, s in reader.scenes()]
        print("[INF] File ",file," for ",subset)
        all_scenes += scene
        for scene_i, (filename, scene_id, paths) in enumerate(scene):
            # Get the trajectories
            raw_traj_data = trajnetplusplustools.Reader.paths_to_xy(paths)
            # Index 0 is the pedestrian of interest
            seq_pos_all.append(raw_traj_data[:,0,:])
            # Displacements
            ped_seq_rel = raw_traj_data[1:,0,:] - raw_traj_data[:-1,0,:]
            seq_rel_all.append(ped_seq_rel)
            # Orientations
            theta_seq_data = np.expand_dims(np.arctan2(raw_traj_data[1:,0,1] - raw_traj_data[:-1,0,1],raw_traj_data[1:,0,0] - raw_traj_data[:-1,0,0]),axis=1)
            seq_theta_all.append(theta_seq_data)
            # Neighbors
            neighbor_paths = raw_traj_data[1:,1:,:]
            neighbor_paths = np.concatenate([neighbor_paths,np.ones([neighbor_paths.shape[0],neighbor_paths.shape[1],1])],axis=2)
            neighbors_n    = neighbor_paths.shape[1]
            if neighbors_n>neighbors_n_max:
                neighbors_n_max = neighbors_n
            seq_neighbors_all.append(neighbor_paths)
            # Social interactions
            of_sim = OpticalFlowSimulator()
            flow,vis_neigh,vis_obst = of_sim.compute_opticalflow_seq(raw_traj_data[1:9,0,:],neighbor_paths[0:8], None)
            all_flow.append(flow)
            all_vis_neigh.append(vis_neigh)

    seq_pos_all      = np.array(seq_pos_all)
    seq_rel_all      = np.array(seq_rel_all)
    seq_theta_all    = np.array(seq_theta_all)
    all_flow         = np.array(all_flow)
    all_vis_neigh    = np.array(all_vis_neigh)
    for i in range(len(seq_neighbors_all)):
        # TODO: avoid using 3 dimensions
        tmp  =np.NaN*np.ones([seq_neighbors_all[i].shape[0],neighbors_n_max,3])
        tmp[:,:seq_neighbors_all[i].shape[1],:]=seq_neighbors_all[i]
        seq_neighbors_all[i]=tmp
    seq_neighbors_all=  np.array(seq_neighbors_all)
    print("[INF] Total trajectories: ",seq_pos_all.shape[0])
    # We get the obs traj and pred_traj
    # [total, obs_len, 2]
    # [total, pred_len, 2]
    obs_traj      = seq_pos_all[:, 1:9, :]
    obs_traj_theta= seq_theta_all[:, 1:9, :]
    pred_traj     = seq_pos_all[:, 9:, :]
    obs_traj_rel  = seq_rel_all[:, :8, :]
    pred_traj_rel = seq_rel_all[:, 8:, :]
    neighbors_obs = seq_neighbors_all[:, 1:9, :]
    # Save all these data as a dictionary
    data = {
        "obs_traj": obs_traj,
        "obs_traj_rel": obs_traj_rel,
        "obs_traj_theta":obs_traj_theta,
        "pred_traj": pred_traj,
        "pred_traj_rel": pred_traj_rel,
        "obs_neighbors": neighbors_obs,
        "obs_optical_flow": all_flow,
        "obs_visible_neighbors": all_vis_neigh
    }
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=25, type=int, help='number of epochs')
    parser.add_argument('--save_every', default=5, type=int,help='frequency of saving model (in terms of epochs)')
    parser.add_argument('--obs_length', default=9, type=int,help='observation length')
    parser.add_argument('--pred_length', default=12, type=int,help='prediction length')
    parser.add_argument('--start_length', default=0, type=int,help='starting time step of encoding observation')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=1e-3, type=float,help='initial learning rate')
    parser.add_argument('--path', default='trajdata',help='glob expression for data files')
    args = parser.parse_args()


    ## Prepare data
    train_data = prepare_data_trajnetplusplus(args.path, subset='/train/')
    val_data   = prepare_data_trajnetplusplus(args.path, subset='/val/')
    print("[INF] Total number of training trajectories:",train_data["obs_traj"].shape[0])
    print("[INF] Total number of validation trajectories:",val_data["obs_traj"].shape[0])

    # Select a random sequence within this dataset
    idSample = random.sample(range(1,train_data["obs_traj"].shape[0]), 1)
    # The random sequence selected
    traj_sample             = train_data['obs_traj'][idSample][0]
    traj_neighbors          = train_data['obs_neighbors'][idSample][0]
    optical_flow_sample     = train_data['obs_optical_flow'][idSample][0]
    visible_neighbors_sample= train_data['obs_visible_neighbors'][idSample][0]
    OFSimulator          = OpticalFlowSimulator()
    # Plot simulated optical flow
    OFSimulator.plot_flow(traj_sample,traj_neighbors,optical_flow_sample,visible_neighbors_sample,None,None,title="Sample optical flow")

if __name__ == '__main__':
    main()
