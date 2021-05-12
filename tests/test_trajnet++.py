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
import pickle

def prepare_data_trajnetplusplus(datasets_path, datasets_names,parameters):
    """ Prepares the train/val scenes and corresponding goals
    Parameters
    ----------
    parameters: Experiment_Parameters
        Defines the prediction experiment parameters.
    path:
        Path to the dataset (set of json files)

    Returns
    -------
    data : Dictionary
        Contains the different processed data as numpy nd arrays
    """
    all_ped_traj_abs      = []
    all_ped_traj_rel      = []
    all_ped_traj_theta    = []
    all_neigbors_traj_abs = []
    all_flows             = []
    all_visible_neighbors = []
    neighbors_n_max       = 0
    # Optical flow
    of_sim = OpticalFlowSimulator()
    ## Iterate over file names
    for dataset_name in datasets_names:
        reader = trajnetplusplustools.Reader(datasets_path + dataset_name + '.ndjson', scene_type='paths')
        ## Necessary modification of train scene to add filename
        scene = [(dataset_name, s_id, s) for s_id, s in reader.scenes()]
        print("[INF] File",dataset_name,"with",len(scene),"scenes.")
        for scene_i, (filename, scene_id, paths) in enumerate(scene):
            # Get the trajectories
            raw_traj_abs = trajnetplusplustools.Reader.paths_to_xy(paths)
            ped_traj_abs = raw_traj_abs[:,0,:]
            # Keep the full trajectory of the pedestrian of interest (start at 0)
            all_ped_traj_abs.append(ped_traj_abs)
            # Displacements along the trajectories (start at 1)
            ped_traj_rel = ped_traj_abs[1:,:] - ped_traj_abs[:-1,:]
            all_ped_traj_rel.append(ped_traj_rel)
            # Orientations
            ped_traj_theta = np.expand_dims(np.arctan2(ped_traj_abs[1:,1]-ped_traj_abs[:-1,1],ped_traj_abs[1:,0]-ped_traj_abs[:-1,0]),axis=1)
            all_ped_traj_theta.append(ped_traj_theta)
            # Neighbors
            neigbors_traj_abs = raw_traj_abs[1:1+parameters.obs_len,1:,:]
            neigbors_traj_abs = np.concatenate([np.ones([neigbors_traj_abs.shape[0],neigbors_traj_abs.shape[1],1]),neigbors_traj_abs],axis=2)
            neighbors_n    = neigbors_traj_abs.shape[1]
            if neighbors_n>neighbors_n_max:
                 neighbors_n_max = neighbors_n
            all_neigbors_traj_abs.append(neigbors_traj_abs)
            flow,vis_neigh,__ = of_sim.compute_opticalflow_seq(ped_traj_abs[1:1+parameters.obs_len,:],neigbors_traj_abs[0:parameters.obs_len,:,:], None)
            all_flows.append(flow)
            all_visible_neighbors.append(vis_neigh)

    all_ped_traj_abs     = np.array(all_ped_traj_abs,dtype='float16')
    all_ped_traj_rel     = np.array(all_ped_traj_rel,dtype='float16')
    all_ped_traj_theta   = np.array(all_ped_traj_theta,dtype='float16')
    all_flows            = np.array(all_flows,dtype='float16')
    all_visible_neighbors= np.array(all_visible_neighbors,dtype='float16')
    for i in range(len(all_neigbors_traj_abs)):
        # TODO: avoid using 3 dimensions?
        tmp=np.NaN*np.ones([all_neigbors_traj_abs[i].shape[0],neighbors_n_max,3])
        tmp[:,:all_neigbors_traj_abs[i].shape[1],:]=all_neigbors_traj_abs[i]
        all_neigbors_traj_abs[i]=tmp
    all_neigbors_traj_abs=  np.array(all_neigbors_traj_abs,dtype='float16')
    print("[INF] Total trajectories: ",all_ped_traj_abs.shape[0])
    # We get the obs traj and pred_traj
    # [total, obs_len, 2]
    # [total, pred_len, 2]
    obs_traj      = all_ped_traj_abs[:,1:1+parameters.obs_len,:]
    obs_traj_theta= all_ped_traj_theta[:,1:1+parameters.obs_len,:]
    pred_traj     = all_ped_traj_abs[:,1+parameters.obs_len:,:]
    obs_traj_rel  = all_ped_traj_rel[:,:parameters.obs_len,:]
    pred_traj_rel = all_ped_traj_rel[:,parameters.obs_len:,:]
    neighbors_obs = all_neigbors_traj_abs[:,:parameters.obs_len,:]
    # Save all these data as a dictionary
    data = {
        "obs_traj": obs_traj,
        "obs_traj_rel": obs_traj_rel,
        "obs_traj_theta":obs_traj_theta,
        "pred_traj": pred_traj,
        "pred_traj_rel": pred_traj_rel,
        "obs_neighbors": neighbors_obs,
        "obs_optical_flow": all_flows,
        "obs_visible_neighbors": all_visible_neighbors
    }
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs_length', default=8, type=int,help='observation length')
    parser.add_argument('--pred_length', default=12, type=int,help='prediction length')
    parser.add_argument('--path', default='trajdata',help='glob expression for data files')
    args     = parser.parse_args()
    datasets = ["biwi_hotel","crowds_students001","crowds_students003","crowds_zara01","crowds_zara03","lcas","wildtrack","cff_06"]
    # Load the default parameters
    experiment_parameters = Experiment_Parameters(add_social=True,add_kp=False)
    experiment_parameters.obs_len  = args.obs_length
    experiment_parameters.pred_len = args.pred_length
    ## Prepare data
    train_data = prepare_data_trajnetplusplus(args.path,datasets,experiment_parameters)
    print("[INF] Total number of training trajectories:",train_data["obs_traj"].shape[0])
    # Training dataset
    pickle_out = open('test.pickle',"wb")
    pickle.dump(train_data, pickle_out, protocol=2)
    pickle_out.close()
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
