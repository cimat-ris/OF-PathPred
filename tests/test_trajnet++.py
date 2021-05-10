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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../trajnet++/trajnetplusplustools')))
import trajnetplusplustools
import logging
import socket


def prepare_data_mine(datasets_path, datasets_names, parameters):
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
    for idx,dataset_name in enumerate(datasets_names):
        seq_neighbors_dataset= []
        seq_ids_dataset      = []
        seq_pos_dataset      = []
        #TODO: avoid having the csv name here
        traj_data_path       = os.path.join(datasets_path+dataset_name, 'mundo/mun_pos.csv')
        print("[INF] Reading "+traj_data_path)
        # Read obstacles files
        if parameters.obstacles:
            print("[INF] Reading obstacle files")
            obstacles_world = load_world_obstacle_polygons(datasets_path,dataset_name)
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

def prepare_data(path, subset='/train/', sample=1.0):
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
    all_scenes   = []
    seq_pos_all  = []
    seq_rel_all  = []
    seq_theta_all= []

    ## List file names
    files = [f.split('.')[-2] for f in os.listdir(path + subset) if f.endswith('.ndjson')]


    ## Iterate over file names
    for file in files:
        reader = trajnetplusplustools.Reader(path + subset + file + '.ndjson', scene_type='paths')
        ## Necessary modification of train scene to add filename
        scene = [(file, s_id, s) for s_id, s in reader.scenes(sample=sample)]
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
            # We suppose that the frame ids are in ascending order
            # frame_ids = np.unique(raw_traj_data[:, 0]).tolist()
            # print("[INF] Total number of frames: ",len(frame_ids))

    seq_pos_all   = np.array(seq_pos_all)
    seq_rel_all   = np.array(seq_rel_all)
    seq_theta_all = np.array(seq_theta_all)
    print("[INF] Total trajectories: ",seq_pos_all.shape[0])
    # We get the obs traj and pred_traj
    # [total, obs_len, 2]
    # [total, pred_len, 2]
    obs_traj      = seq_pos_all[:, 1:9, :]
    obs_traj_theta= seq_theta_all[:, 1:9, :]
    pred_traj     = seq_pos_all[:, 9:, :]
    #frame_obs     = seq_frames_all[:, :obs_len]
    obs_traj_rel  = seq_rel_all[:, :8, :]
    pred_traj_rel = seq_rel_all[:, 8:, :]
    #nedighbors_obs= seq_neighbors_all[:, :obs_len, :]
    # Save all these data as a dictionary
    data = {
        "obs_traj": obs_traj,
        "obs_traj_rel": obs_traj_rel,
        "obs_traj_theta":obs_traj_theta,
        "pred_traj": pred_traj,
        "pred_traj_rel": pred_traj_rel,
        #"frames_ids": frame_obs,
        #"obs_neighbors": neighbors_obs
    }
    #return data
    return all_scenes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=25, type=int, help='number of epochs')
    parser.add_argument('--save_every', default=5, type=int,help='frequency of saving model (in terms of epochs)')
    parser.add_argument('--obs_length', default=9, type=int,help='observation length')
    parser.add_argument('--pred_length', default=12, type=int,help='prediction length')
    parser.add_argument('--start_length', default=0, type=int,help='starting time step of encoding observation')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=1e-3, type=float,help='initial learning rate')
    parser.add_argument('--step_size', default=10, type=int,help='step_size of lr scheduler')
    parser.add_argument('-o', '--output', default=None,help='output file')
    parser.add_argument('--path', default='trajdata',help='glob expression for data files')
    parser.add_argument('--type', default='vanilla',choices=('vanilla', 'occupancy', 'directional', 'social', 'hiddenstatemlp', 's_att_fast','directionalmlp', 'nn', 'attentionmlp', 'nn_lstm', 'traj_pool', 'nmmp', 'dir_social'),help='type of interaction encoder')
    parser.add_argument('--sample', default=1.0, type=float,help='sample ratio when loading train/val scenes')

    ## Augmentations
    parser.add_argument('--augment', action='store_true',help='perform rotation augmentation')
    parser.add_argument('--normalize_scene', action='store_true',help='rotate scene so primary pedestrian moves northwards at end of observation')
    parser.add_argument('--augment_noise', action='store_true',help='flag to add noise to observations for robustness')

    ## Sequence Encoder Hyperparameters
    hyperparameters = parser.add_argument_group('hyperparameters')
    hyperparameters.add_argument('--hidden-dim', type=int, default=128,help='LSTM hidden dimension')
    hyperparameters.add_argument('--coordinate-embedding-dim', type=int, default=64,help='coordinate embedding dimension')
    hyperparameters.add_argument('--pool_dim', type=int, default=256, help='output dimension of interaction vector')

    args = parser.parse_args()

    ## Fixed set of scenes if sampling
    if args.sample < 1.0:
        # TODO
        #torch.manual_seed("080819")
        random.seed(1)

    ## Define location to save trained model
    if not os.path.exists('OUTPUT_BLOCK/{}'.format(args.path)):
        os.makedirs('OUTPUT_BLOCK/{}'.format(args.path))
    args.output = 'OUTPUT_BLOCK/{}/vae_{}_{}.pkl'.format(args.path, args.type, args.output)

    # configure logging
    from pythonjsonlogger import jsonlogger
    file_handler = logging.FileHandler(args.output + '.log', mode='w')
    # TODO
    #file_handler.setFormatter(jsonlogger.JsonFormatter('(message) (levelname) (name) (asctime)'))
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler])
    logging.info({
            'type': 'process',
            'argv': sys.argv,
            'args': vars(args),
            'hostname': socket.gethostname(),
    })

    ## Prepare data
    train_scenes = prepare_data(args.path, subset='/train/', sample=args.sample)
    ntrain = 0
    for scene_i, (filename, scene_id, paths) in enumerate(train_scenes):
        scene_start = time.time()
        ## Make new scene
        paths_xy = trajnetplusplustools.Reader.paths_to_xy(paths)
        ntrain += paths_xy.shape[1]
    val_scenes   = prepare_data(args.path, subset='/val/', sample=args.sample)
    nvals = 0
    for scene_i, (filename, scene_id, paths) in enumerate(val_scenes):
        scene_start = time.time()
        ## Make new scene
        paths_xy = trajnetplusplustools.Reader.paths_to_xy(paths)
        nvals += paths_xy.shape[1]
    print("[INF] Total number of training trajectories:",ntrain)
    print("[INF] Total number of validation trajectories:",nvals)

if __name__ == '__main__':
    main()
