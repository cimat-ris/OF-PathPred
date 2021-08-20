import os, glob, sys, logging
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
        reader = Reader(datasets_path + dataset_name + '.ndjson', scene_type='paths')
        ## Necessary modification of train scene to add filename
        scene = [(dataset_name, s_id, s) for s_id, s in reader.scenes()]
        logging.info("File "+dataset_name+" with {} scenes.".format(len(scene)))
        for scene_i, (filename, scene_id, paths) in enumerate(scene):
            # Get the trajectories
            raw_traj_abs = Reader.paths_to_xy(paths)
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
            if keep_neighbors:
                neighbors_n    = neigbors_traj_abs.shape[1]
                if neighbors_n>neighbors_n_max:
                    neighbors_n_max = neighbors_n
                all_neigbors_traj_abs.append(neigbors_traj_abs)
            # Optical flow
            flow,vis_neigh,__ = of_sim.compute_opticalflow_seq(ped_traj_abs[1:1+parameters.obs_len,:],neigbors_traj_abs[0:parameters.obs_len,:,:], None)
            all_flows.append(flow)
            all_visible_neighbors.append(vis_neigh)

    all_ped_traj_abs     = np.array(all_ped_traj_abs,dtype='float16')
    all_ped_traj_rel     = np.array(all_ped_traj_rel,dtype='float16')
    all_ped_traj_theta   = np.array(all_ped_traj_theta,dtype='float16')
    all_flows            = np.array(all_flows)
    all_visible_neighbors= np.array(all_visible_neighbors,dtype='float16')

    # Data sanity check
    logging.debug("Checking data consistency")
    logging.debug("Nan in all_ped_traj_abs {} ".format(np.isnan(all_ped_traj_abs).any()))
    logging.debug("Nan in all_ped_traj_rel {} ".format(np.isnan(all_ped_traj_rel).any()))
    logging.debug("Nan in all_ped_traj_theta {} ".format(np.isnan(all_ped_traj_theta).any()))
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
        all_neigbors_traj_abs=  np.array(all_neigbors_traj_abs,dtype='float16')
    logging.info("Total trajectories: {}".format(all_ped_traj_abs.shape[0]))
    # We get the obs traj and pred_traj
    # [total, obs_len, 2]
    # [total, pred_len, 2]
    obs_traj      = all_ped_traj_abs[:,1:1+parameters.obs_len,:]
    obs_traj_theta= all_ped_traj_theta[:,1:1+parameters.obs_len,:]
    pred_traj     = all_ped_traj_abs[:,1+parameters.obs_len:,:]
    obs_traj_rel  = all_ped_traj_rel[:,:parameters.obs_len,:]
    pred_traj_rel = all_ped_traj_rel[:,parameters.obs_len:,:]
    if keep_neighbors:
        neighbors_obs = all_neigbors_traj_abs[:,:parameters.obs_len,:]
    # Save all these data as a dictionary
    data = {
        "obs_traj": obs_traj,
        "obs_traj_rel": obs_traj_rel,
        "obs_traj_theta":obs_traj_theta,
        "obs_optical_flow": all_flows,
        "obs_visible_neighbors": all_visible_neighbors,
        "pred_traj": pred_traj,
        "pred_traj_rel": pred_traj_rel
    }
    if keep_neighbors:
        data["obs_neighbors"]        = neighbors_obs
    return data

def prepare_data(datasets_path, datasets_names, parameters):
    datasets = range(len(datasets_names))
    datasets = list(datasets)

    # Paths for the datasets used to form the training set
    used_data_dirs = [datasets_names[x] for x in datasets]
    # Sequence lengths
    obs_len  = parameters.obs_len
    pred_len = parameters.pred_len
    seq_len  = obs_len + pred_len
    logging.info("Sequence length (observation+prediction): {}".format(seq_len))

    # Lists that will hold the data
    num_person_starting_at_frame = []
    seq_pos_all                  = []
    seq_neighbors_all            = []
    seq_frames_all               = []  # [N, seq_len]
    all_flow                     = []
    all_vis_neigh                = []
    all_vis_obst                 = []
    # Scan all the datasets
    for idx,dataset_name in enumerate(datasets_names):
        seq_neighbors_dataset= []
        seq_pos_dataset      = []
        #TODO: avoid having the csv name here
        traj_data_path       = os.path.join(datasets_path+dataset_name, 'mundo/mun_pos.csv')
        logging.info("Reading "+traj_data_path)
        # Read obstacles files
        if parameters.obstacles:
            logging.info("Reading obstacle files")
            obstacles_world = load_world_obstacle_polygons(datasets_path,dataset_name)
        else:
            obstacles_world = None

        # Raw trayectory coordinates
        raw_traj_data = np.genfromtxt(traj_data_path, delimiter= parameters.delim)

        # We suppose that the frame ids are in ascending order
        frame_ids = np.unique(raw_traj_data[:, 0]).tolist()
        logging.info("Total number of frames: {}".format(len(frame_ids)))

        raw_traj_data_per_frame = [] # people in frame
        # Group the spatial pedestrian data frame by frame
        # List indexed by frame ids.
        # Data: id_frame, id_person, x, y
        for frame in frame_ids:
            raw_traj_data_per_frame.append(raw_traj_data[raw_traj_data[:, 0]==frame, :])

        # Iterate over the frames
        for idx, frame in enumerate(frame_ids):
            # Consider frame sequences of size seq_len = obs+pred
            # id_frame, id_person, x, y por every person present in the frame
            raw_seq_data = raw_traj_data_per_frame[idx:idx+seq_len]
            raw_seq_data = np.concatenate(raw_seq_data,axis=0)
            # Unique indices for the persons in the sequence "raw_seq_data"
            peds_in_seq = np.unique(raw_seq_data[:,1])
            # Number of pedestrians to consider
            num_peds_in_seq = peds_in_seq.shape[0]
            # The following two arrays have the same shape
            # "pos_seq_data" contains all the absolute positions of all the pedestrians in the sequence
            # and the information is encoded in an absolute frame (no transformation)
            pos_seq_data = np.zeros((num_peds_in_seq, seq_len, 2), dtype="float32")
            # Is the array that have the sequence of Id_person of all people that there are in frame sequence
            frame_ids_seq_data = np.zeros((num_peds_in_seq, seq_len), dtype="int32")

            # Maximum number of persons in a frame
            person_max = parameters.person_max
            # Absolute pixel-based data: id_person, x, y
            neighbors_data = np.zeros((num_peds_in_seq, seq_len, person_max, 3),dtype="float32")
            neighbors_data[:] = np.NaN

            ped_count = 0
            # For all the persons appearing in this sequence
            # We will make one entry in the sequences list provided that it is present in all frames
            for ped_id in peds_in_seq:
                # Get the information about ped_id, in the whole sequence
                ped_seq_data = raw_seq_data[raw_seq_data[:,1]==ped_id,:]
                # We want pedestrians whose number of observations inside this sequence is exactly seq_len
                if len(ped_seq_data) != seq_len:
                    # We do not have enough observations for this person
                    continue

                # Check whether the first 8 positions are not the same
                # TODO: is it OK to do that?
                equal_consecutive = 0
                for n in range(obs_len-1):
                    if((ped_seq_data[n,2]==ped_seq_data[n+1,2]) and (ped_seq_data[n,3]==ped_seq_data[n+1,3])):
                        equal_consecutive +=1
                if(equal_consecutive==obs_len-1):
                    continue

                # To keep neighbors data for the person ped_id
                neighbors_ped_seq = np.zeros((seq_len, person_max, 3),dtype="float32")
                # Scan all the frames of the sequence
                for frame_idx,frame_id in enumerate(np.unique(raw_seq_data[:,0]).tolist()):
                    # Information of frame "frame_id"
                    frame_data = raw_seq_data[raw_seq_data[:,0]==frame_id,:]
                    # Id, x, y of the pedestrians of frame "frame_id"
                    frame_data = frame_data[:,1:4]
                    # For all the persons in the sequence
                    for neighbor_ped_idx,neighbor_ped_id in enumerate(peds_in_seq):
                        # Get the data of this specific person
                        neighbor_data = frame_data[frame_data[:, 0]==neighbor_ped_id,:]
                        # If we have information for this pedestrian, add it to the neighbors struture
                        if neighbor_data.size != 0 and neighbor_ped_idx<person_max:
                            neighbors_ped_seq[frame_idx,neighbor_ped_idx,:] = neighbor_data
                # Contains the neighbor data for ped_count
                neighbors_data[ped_count,:,:,:] = neighbors_ped_seq

                # Spatial data (absolute positions) for ped
                ped_seq_pos = ped_seq_data[:,2:]
                # Absolute x,y and displacements for all person_id
                pos_seq_data[ped_count, :, :] = ped_seq_pos
                # For each tracked person
                # we keep the list of all the frames in which it is present
                frame_ids_seq_data[ped_count, :] = frame_ids[idx:idx+seq_len]
                # Increment ped_count (persons )
                ped_count += 1

            # Number of persons getting a sequence starting at this frame
            num_person_starting_at_frame.append(ped_count)
            # Only count_ped data are preserved in the following three arrays
            # Append all the trajectories (pos_seq_data) starting at this frame
            seq_pos_all.append(pos_seq_data[:ped_count])
            # Append all the frame ranges (frame_ids_seq_data) starting at this frame
            seq_frames_all.append(frame_ids_seq_data[:ped_count])
            # Information used locally for this dataset
            seq_pos_dataset.append(pos_seq_data[:ped_count])
            # Neighbours
            seq_neighbors_all.append(neighbors_data[:ped_count])
            # Append all the neighbor data (neighbors_data) starting at this frame
            seq_neighbors_dataset.append(neighbors_data[:ped_count])

        # Neighbors information
        seq_neighbors_dataset = np.concatenate(seq_neighbors_dataset, axis = 0)
        obs_neighbors         = seq_neighbors_dataset[:,:obs_len,:,:]
        seq_pos_dataset = np.concatenate(seq_pos_dataset,axis=0)
        obs_traj        = seq_pos_dataset[:, :obs_len, :]
        logging.info("Total number of trajectories in this dataset: {}".format(obs_traj.shape[0]))
        # At the dataset level
        logging.info("Add social interaction data (optical flow)")
        if parameters.obstacles:
            of_sim = OpticalFlowSimulator()
            flow,vis_neigh,vis_obst = of_sim.compute_opticalflow_batch(obs_neighbors, obs_traj,parameters.obs_len,obstacles_world)
        else:
            of_sim = OpticalFlowSimulator()
            flow,vis_neigh,vis_obst = of_sim.compute_opticalflow_batch(obs_neighbors, obs_traj,parameters.obs_len,None)
        all_flow.append(flow)
        all_vis_neigh.append(vis_neigh)
        all_vis_obst.append(vis_obst)
    # Upper level (all datasets)
    # Concatenate all the content of the lists (pos/relative pos/frame ranges)
    seq_pos_all   = np.concatenate(seq_pos_all, axis=0)
    # TODO: rel and theta could be simply generated here
    # All the displacements are estimated here.
    seq_rel_all           = np.zeros_like(seq_pos_all)
    seq_rel_all[:,1:,:]   = seq_pos_all[:,1:,:]-seq_pos_all[:,:-1,:]
    # Note that padding is done at the first displacement with the second displacement
    seq_rel_all[:,0,:]    = seq_rel_all[:,1,:]
    seq_theta_all         = np.zeros_like(seq_pos_all[:,:,0:1])
    seq_theta_all[:,1:,0] = np.arctan2(seq_pos_all[:, 1:, 1] - seq_pos_all[:,:-1, 1],seq_pos_all[:,1:, 0] - seq_pos_all[:,:-1, 0])
    seq_theta_all[:,0,  0]= seq_theta_all[:,1,  0]

    seq_frames_all    = np.concatenate(seq_frames_all, axis=0)
    seq_neighbors_all = np.concatenate(seq_neighbors_all, axis=0)
    logging.info("Total number of sample sequences: ".format(len(seq_pos_all)))

    # We get the obs traj and pred_traj
    # [total, obs_len, 2]
    # [total, pred_len, 2]
    obs_traj      = seq_pos_all[:, :obs_len, :]
    obs_traj_theta= seq_theta_all[:, :obs_len, :]
    pred_traj     = seq_pos_all[:, obs_len:, :]
    frame_obs     = seq_frames_all[:, :obs_len]
    obs_traj_rel  = seq_rel_all[:, :obs_len, :]
    pred_traj_rel = seq_rel_all[:, obs_len:, :]
    neighbors_obs= seq_neighbors_all[:, :obs_len, :]
    # Save all these data as a dictionary
    data = {
        "obs_traj": obs_traj,
        "obs_traj_rel": obs_traj_rel,
        "obs_traj_theta":obs_traj_theta,
        "pred_traj": pred_traj,
        "pred_traj_rel": pred_traj_rel,
        "frames_ids": frame_obs,
        "obs_neighbors": neighbors_obs
    }

    # Optical flow
    all_flow     = np.concatenate(all_flow,axis=0)
    all_vis_neigh= np.concatenate(all_vis_neigh,axis=0)
    data.update({
        "obs_optical_flow": all_flow,
        "obs_visible_neighbors": all_vis_neigh
    })
    if parameters.obstacles:
        all_vis_obst = np.concatenate(all_vis_obst,axis=0)
        data.update({
                "obs_visible_obstacles": all_vis_obst})
    return data
