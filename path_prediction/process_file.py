import os
from tqdm import tqdm
import glob
import numpy as np
from path_prediction.interaction_optical_flow import OpticalFlowSimulator
from path_prediction.obstacles import load_world_obstacle_polygons

def prepare_data(path, subset='/train/', sample=1.0, goals=True):
    """ Prepares the train/val scenes and corresponding goals

    Parameters
    ----------
    subset: String ['/train/', '/val/']
        Determines the subset of data to be processed
    sample: Float (0.0, 1.0]
        Determines the ratio of data to be sampled
    goals: Bool
        If true, the goals of each track are extracted
        The corresponding goal file must be present in the 'goal_files' folder
        The name of the goal file must be the same as the name of the training file

    Returns
    -------
    all_scenes: List
        List of all processed scenes
    all_goals: Dictionary
        Dictionary of goals corresponding to each dataset file.
        None if 'goals' argument is False.
    """

    ## read goal files
    all_goals = {}
    all_scenes = []

    ## List file names
    files = [f.split('.')[-2] for f in os.listdir(path + subset) if f.endswith('.ndjson')]
    ## Iterate over file names
    for file in files:
        print("Reading file ",file," for ",subset)
        reader = trajnetplusplustools.Reader(path + subset + file + '.ndjson', scene_type='paths')
        ## Necessary modification of train scene to add filename
        scene = [(file, s_id, s) for s_id, s in reader.scenes(sample=sample)]
        print("")
        print(scene[0])
        if goals:
            goal_dict = pickle.load(open('goal_files/' + subset + file +'.pkl', "rb"))
            ## Get goals corresponding to train scene
            all_goals[file] = {s_id: [goal_dict[path[0].pedestrian] for path in s] for _, s_id, s in scene}
        all_scenes += scene

    if goals:
        return all_scenes, all_goals
    return all_scenes, None


def process_file(datasets_path, datasets_names, parameters, csv_file='mundo/mun_pos.csv'):
    datasets = range(len(datasets_names))
    datasets = list(datasets)

    # Paths for the datasets used to form the training set
    used_data_dirs = [datasets_names[x] for x in datasets]

    # Sequence lengths
    obs_len  = parameters.obs_len
    pred_len = parameters.pred_len
    seq_len  = obs_len + pred_len
    print("[INF] Sequence length (observation+prediction):", seq_len)

    # Lists that will hold the data
    num_person_starting_at_frame = []
    seq_pos_all                  = []
    seq_theta_all                = []
    seq_rel_all                  = []
    seq_neighbors_all            = []
    seq_frames_all               = []  # [N, seq_len]
    all_flow                     = []
    all_vis_neigh                = []
    all_vis_obst                 = []
    # Scan all the datasets
    for idx,directory in enumerate(used_data_dirs):
        seq_neighbors_dataset= []
        seq_ids_dataset      = []
        seq_pos_dataset      = []
        #TODO: avoid having the csv name here
        if len(csv_file)>1:
            traj_data_path       = os.path.join(datasets_path+directory, csv_file)
        else:
            traj_data_path       = datasets_path+directory
        print("[INF] Reading "+traj_data_path)
        # Read obstacles files
        if parameters.obstacles:
            print("[INF] Reading obstacle files")
            t = directory.split('/')
            data_paths = t[0]+'/'+t[1]+'/'
            dataset_name = t[2]
            obstacles_world = load_world_obstacle_polygons(data_paths,dataset_name)
        else:
            obstacles_world = None

        # Raw trayectory coordinates
        raw_traj_data = np.genfromtxt(traj_data_path, delimiter= parameters.delim)

        # We suppose that the frame ids are in ascending order
        frame_ids = np.unique(raw_traj_data[:, 0]).tolist()
        print("[INF] Total number of frames: ",len(frame_ids))

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
            if parameters.intersection:
                # Intersection of the id_person of "raw_seq_data"
                # i.e. the pedestrian ids present in ALL th frames
                peds_in_seq = reduce(set.intersection,
                                [set(peds_id_list[:,1]) for peds_id_list in
                                raw_seq_data])
                peds_in_seq = sorted(list(peds_in_seq))
                raw_seq_data= np.concatenate(raw_seq_data,axis=0)
            else:
                raw_seq_data= np.concatenate(raw_seq_data,axis=0)
                # Unique indices for the persons in the sequence "raw_seq_data"
                peds_in_seq = np.unique(raw_seq_data[:,1])
                # List of all the persons in this sequence
                peds_in_seq = list(peds_in_seq)
            # Number of pedestrians to consider
            num_peds_in_seq = len(peds_in_seq)

            # The following two arrays have the same shape
            # "pos_seq_data" contains all the absolute positions of all the pedestrians in the sequence
            # and he information is encoded in an absolute frame (no transformation)
            pos_seq_data = np.zeros((num_peds_in_seq, seq_len, 2), dtype="float32")
            # Same, with only the displacements
            rel_seq_data = np.zeros((num_peds_in_seq, seq_len, 2), dtype="float32")
            # Same with orientations
            theta_seq_data = np.zeros((num_peds_in_seq, seq_len, 1), dtype="float32")
            # Is the array that have the sequence of Id_person of all people that there are in frame sequence
            frame_ids_seq_data = np.zeros((num_peds_in_seq, seq_len), dtype="int32")

            # Maximum number of persons in a frame
            person_max = parameters.person_max
            # Absolute pixel-based data: id_person, x, y
            neighbors_data = np.zeros((num_peds_in_seq, seq_len, person_max, 3),dtype="float32")

            ped_count = 0
            # For all the persons appearing in this sequence
            # We will make one entry in the sequences list
            for ped_id in peds_in_seq:
                # Get the information about ped_id, in the whole sequence
                ped_seq_data = raw_seq_data[raw_seq_data[:,1]==ped_id,:]
                # We want pedestrians whose number of observations inside this sequence is exactly seq_len
                if len(ped_seq_data) != seq_len:
                    # We do not have enough observations for this person
                    continue

                # List of all the persons in the frame, to build the neighbors array
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
                        if neighbor_data.size != 0:
                            neighbors_ped_seq[frame_idx,neighbor_ped_idx,:] = neighbor_data
                # Contains the neighbor data for ped_count
                neighbors_data[ped_count,:,:,:] = neighbors_ped_seq

                # Spatial data (absolute positions) for ped
                ped_seq_pos = ped_seq_data[:,2:]
                # Spatial data (relative)
                ped_seq_rel = np.zeros_like(ped_seq_pos)
                if parameters.output_representation=='dxdy':
                    # First frame of the relative array is set to zeros
                    ped_seq_rel[1:, :] = ped_seq_pos[1:, :] - ped_seq_pos[:-1, :]
                else:
                    ped_seq_rel[1:, 0] = np.log(0.001+np.linalg.norm(ped_seq_pos[1:, :] - ped_seq_pos[:-1, :],axis=1)/0.5)
                    ped_seq_rel[1:, 1] = (np.arctan2(ped_seq_pos[1:, 1] - ped_seq_pos[:-1, 1],ped_seq_pos[1:, 0] - ped_seq_pos[:-1, 0]))/10.0
                    ped_seq_rel[0, :]  = ped_seq_rel[1, :]

                # Absolute x,y and displacements for all person_id
                pos_seq_data[ped_count, :, :] = ped_seq_pos
                rel_seq_data[ped_count, :, :] = ped_seq_rel
                # Orientations
                theta_seq_data[ped_count,1:, 0] = np.arctan2(ped_seq_pos[1:, 1] - ped_seq_pos[:-1, 1],ped_seq_pos[1:, 0] - ped_seq_pos[:-1, 0])
                theta_seq_data[ped_count,0,  0] = theta_seq_data[ped_count,1,  0]

                # For each tracked person
                # we keep the list of all the frames in which it is present
                frame_ids_seq_data[ped_count, :] = frame_ids[idx:idx+seq_len]
                # List of persons TODO?
                seq_ids_dataset.append(ped_id)
                # Increment ped_count (persons )
                ped_count += 1

            # Number of persons getting a sequence starting at this frame
            num_person_starting_at_frame.append(ped_count)
            # Only count_ped data are preserved in the following three arrays
            # Append all the trajectories (pos_seq_data) starting at this frame
            seq_pos_all.append(pos_seq_data[:ped_count])
            # Append all the displacement trajectories (pos_seq_data) starting at this frame
            seq_rel_all.append(rel_seq_data[:ped_count])
            seq_theta_all.append(theta_seq_data[:ped_count])
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
        vec = {
            "obs_neighbors": obs_neighbors,
            "key_idx": np.array(seq_ids_dataset),
            "obs_traj":  obs_traj
        }
        print("[INF] Total number of trajectories in this dataset: ",obs_traj.shape[0])
        # At the dataset level
        if parameters.add_social:
            print("[INF] Add social interaction data (optical flow)")
            if parameters.obstacles:
                of_sim = OpticalFlowSimulator()
                flow,vis_neigh,vis_obst = of_sim.compute_opticalflow_batch(vec['obs_neighbors'], vec['key_idx'], vec['obs_traj'],parameters.obs_len,obstacles_world)
            else:
                of_sim = OpticalFlowSimulator()
                flow,vis_neigh,vis_obst = of_sim.compute_opticalflow_batch(vec['obs_neighbors'], vec['key_idx'], vec['obs_traj'],parameters.obs_len,None)
            all_flow.append(flow)
            all_vis_neigh.append(vis_neigh)
            all_vis_obst.append(vis_obst)
    # Upper level (all datasets)
    # Concatenate all the content of the lists (pos/relative pos/frame ranges)
    seq_pos_all   = np.concatenate(seq_pos_all, axis=0)
    seq_rel_all   = np.concatenate(seq_rel_all, axis=0)
    seq_theta_all = np.concatenate(seq_theta_all, axis=0)
    seq_frames_all= np.concatenate(seq_frames_all, axis=0)
    seq_neighbors_all = np.concatenate(seq_neighbors_all, axis=0)
    print("[INF] Total number of sample sequences: ",len(seq_pos_all))

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
    if parameters.add_social:
        all_flow     = np.concatenate(all_flow,axis=0)
        all_vis_neigh= np.concatenate(all_vis_neigh,axis=0)
        data.update({
            "obs_optical_flow": all_flow,
            "obs_visible_neighbors": all_vis_neigh
        })
        if parameters.obstacles:
            all_vis_obst = np.concatenate(all_vis_obst,axis=0)
            data.update({
                "obs_visible_obstacles": all_vis_obst
            })
    return data
