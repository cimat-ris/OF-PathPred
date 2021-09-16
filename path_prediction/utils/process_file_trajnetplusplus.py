import os, glob, sys, logging, math
from tqdm import tqdm
import numpy as np
from .interaction_optical_flow import OpticalFlowSimulator
from .obstacles import load_world_obstacle_polygons
# Since it is used as a submodule, the trajnetplusplustools directory should be there
sys.path.append("../../trajnetplusplustools")
from trajnetplusplustools import Reader

def prepare_data_trajnetplusplus(datasets_path, datasets_names,parameters,keep_neighbors=True):
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
    all_neigbors_traj_abs = []
    all_flows             = []
    all_visible_neighbors = []
    neighbors_n_max       = 0
    primary_path = []
    # Optical flow
    of_sim = OpticalFlowSimulator()
    ## Iterate over file names
    for dataset_name in datasets_names:
        reader = Reader(datasets_path + dataset_name + '.ndjson', scene_type='paths')
        ## Necessary modification of train scene to add filename
        scene = [(dataset_name, s_id, s) for s_id, s in reader.scenes()]
        logging.info("File "+dataset_name+" with {} scenes.".format(len(scene)))
        for scene_i, (filename, scene_id, paths) in enumerate(scene):
            # Get the trajectories
            raw_traj_abs = Reader.paths_to_xy(paths)
            ped_traj_abs = raw_traj_abs[:,0,:]
            if ped_traj_abs.shape[0]<1+parameters.obs_len+parameters.pred_len:
                continue
            # Keep the full trajectory of the pedestrian of interest (start at 0)
            all_ped_traj_abs.append(ped_traj_abs)
            # Save info path scene scene_id
            primary_path.append((scene_id, paths[0],reader.scenes_by_id[scene_id]))
            # Neighbors
            neigbors_traj_abs = raw_traj_abs[1:1+parameters.obs_len,1:,:]
            neigbors_traj_abs = np.concatenate([np.ones([neigbors_traj_abs.shape[0],neigbors_traj_abs.shape[1],1]),neigbors_traj_abs],axis=2)
            if keep_neighbors:
                neighbors_n    = neigbors_traj_abs.shape[1]
                if neighbors_n>neighbors_n_max:
                    neighbors_n_max = neighbors_n
                all_neigbors_traj_abs.append(neigbors_traj_abs)
            # Optical flow
            flow,vis_neigh,__ = of_sim.compute_opticalflow_seq(ped_traj_abs[1:1+parameters.obs_len,:],neigbors_traj_abs[0:parameters.obs_len,:,:], None)
            all_flows.append(flow)
            all_visible_neighbors.append(vis_neigh)

    all_ped_traj_abs     = np.array(all_ped_traj_abs, dtype="float32")
    all_flows            = np.array(all_flows, dtype="float32")
    all_visible_neighbors= np.array(all_visible_neighbors)

    # Data sanity check
    logging.debug("Checking data consistency")
    logging.debug("Nan in all_ped_traj_abs {} ".format(np.isnan(all_ped_traj_abs).any()))
    logging.debug("Nan in all_flows {} ".format(np.isnan(all_flows).any()))
    logging.debug("Inf in all_flows {} ".format(np.isinf(all_flows).any()))
    logging.debug("Nan in all_visible_neighbors {} ".format(np.isnan(all_visible_neighbors).any()))
    logging.debug("Inf in all_visible_neighbors {} ".format(np.isinf(all_visible_neighbors).any()))

    if keep_neighbors:
        for i in range(len(all_neigbors_traj_abs)):
            # TODO: avoid using 3 dimensions?
            tmp=np.NaN*np.ones([all_neigbors_traj_abs[i].shape[0],neighbors_n_max,3])
            tmp[:,:all_neigbors_traj_abs[i].shape[1],:]=all_neigbors_traj_abs[i]
            all_neigbors_traj_abs[i]=tmp
        all_neigbors_traj_abs=  np.array(all_neigbors_traj_abs)
    logging.info("Total trajectories: {}".format(all_ped_traj_abs.shape[0]))


    # By broadcasting, center these data
    seq_pos_centered_all  = all_ped_traj_abs - all_ped_traj_abs[:,parameters.obs_len:parameters.obs_len+1,0:2]
    # Displacements
    seq_rel_all           = np.zeros_like(all_ped_traj_abs)
    seq_rel_all[:,1:,:]   = all_ped_traj_abs[:,1:,:]-all_ped_traj_abs[:,:-1,:]
    # All directions
    seq_theta_all         = np.zeros_like(all_ped_traj_abs[:,:,0:1])
    seq_theta_all[:,:,0]  = np.arctan2(seq_rel_all[:,:,1],seq_rel_all[:,:,0])
    # Cosine and sine of the orientation angle at the last observed point
    costheta              = np.cos(seq_theta_all[:,parameters.obs_len:parameters.obs_len+1,0:1])
    sintheta              = np.sin(seq_theta_all[:,parameters.obs_len:parameters.obs_len+1,0:1])
    seq_pos_rot_all       = np.zeros_like(all_ped_traj_abs)
    seq_pos_rot_all[:,:,0:1]= costheta*(seq_pos_centered_all[:,:,0:1])+sintheta*(seq_pos_centered_all[:,:,1:2])
    seq_pos_rot_all[:,:,1:2]=-sintheta*(seq_pos_centered_all[:,:,0:1])+costheta*(seq_pos_centered_all[:,:,1:2])
    # All the displacements are estimated here.
    seq_rel_rot_all         = np.zeros_like(seq_pos_rot_all)
    seq_rel_rot_all[:,1:,:] = seq_pos_rot_all[:,1:,:]-seq_pos_rot_all[:,:-1,:]
    # Save all these data as a dictionary
    data = {
        "obs_traj": all_ped_traj_abs[:,1:1+parameters.obs_len,:],
        "obs_traj_rel": seq_rel_all[:,1:1+parameters.obs_len,:],
        "obs_traj_theta":seq_theta_all[:,1:1+parameters.obs_len,:],
        "obs_optical_flow": all_flows[:,1:1+parameters.obs_len,:],
        "obs_visible_neighbors": all_visible_neighbors[:,1:1+parameters.obs_len,:],
        "pred_traj": all_ped_traj_abs[:,1+parameters.obs_len:,:],
        "pred_traj_rel": seq_rel_all[:,1+parameters.obs_len:,:],
        "index": np.array(range(len(primary_path)))
    }
    if keep_neighbors:
        data["obs_neighbors"]        = all_neigbors_traj_abs[:,1:parameters.obs_len+1,:]
    return data, primary_path
