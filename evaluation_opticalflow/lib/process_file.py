import os
from tqdm import tqdm
import glob
import numpy as np
from interaction_optical_flow import OpticalFlowSimulator
from obstacles import load_world_obstacle_polygons

# Predictor parameters
#Este es cuando solo se procesa un solo archivo y sin interseccion
class predictor_parameters:
    def __init__(self,add_social=False):
        # Maximum number of persons in a frame
        self.person_max = 42   # eth-Univ: 42  eth-Hotel: 28
        # Observation length (trajlet size)
        self.obs_len    = 8
        # Prediction length
        self.pred_len   = 12
        # Flag to consider social interactions
        self.add_social = add_social
        # Number of key points
        self.kp_num     = 18
        # Key point flag
        self.add_kp     = False
        # Obstacles flag
        self.obstacles    = False
        self.neighborhood = False
        self.intersection = False

#Este es cuando se procesa varios archivos y sin interseccion
class predictor_parameters_varios:
    def __init__(self,add_social=False):
        # Maximum number of persons in a frame
        # Is the person max in test set ("example Ucy-Zara02")
        self.person_max = 26 # Eth-Univ: 42  Eth-Hotel: 28  Ucy-Zara02: 26
        # Indice of the test set in data_dirs array
        #data_dirs = ['../data1/eth-univ', '../data1/eth-hotel',
        #     '../data1/ucy-zara01', '../data1/ucy-zara02',
        #     '../data1/ucy-univ']

        self.ind_test   = 3
        # Observation length (trajlet size)
        self.obs_len    = 8
        # Prediction length
        self.pred_len   = 12
        # Flag to consider social interactions
        self.add_social = add_social
        # Number of key points
        self.kp_num     = 18
        # Key point flag
        self.add_kp     = False
        # Obstacles flag
        self.obstacles    = False
        self.neighborhood = False
        self.intersection = False



# En todas estas funciones cuando se usa el modo add_social
# se contempla a los vecinos que pueden no permanecer en toda la secuencia
# y tambien cuando solo nos quedamos con los que siempre permanecen

def process_file(directory, args, delim):

    obs_len  = args.obs_len
    pred_len = args.pred_len
    seq_len  = obs_len + pred_len

    print("[INF] Sequence length (observation+prediction):", seq_len)
    num_person_in_start_frame = []

    seq_list_pos   = []
    seq_list_rel   = []
    seq_list_frames= []
    kp_list      = []  # [N, seq_len, 18, 3]
    kp_list_rel  = []
    all_flow     = []
    # Information about all the sequences with size seq_len
    seq_list_person     = []
    seq_list_person_rel = []

    # Load other features if necessary
    # This list holds the person ids for all the persons of any sequence of length seq_len
    key_idx = []
    kp_feats = {}  # "frameidx_personId"
    #  To use keypoints, we open then from a file
    if args.add_kp:
        kp_path = os.path.join(directory,'kp_box.csv')
        print(kp_path)
        with open(kp_path, "r") as f:
            for line in f:
                fidxykp = line.strip().split(delim)
                key = fidxykp[0] + "_" +fidxykp[1]
                kp_feats[key] = np.array(fidxykp[2:]).reshape(args.kp_num,3)
    # Read obstacles
    if args.obstacles:
        print("[INF] Reading obstacle files")
        t = directory.split('/')
        data_paths = t[0]+'/'+t[1]+'/'
        dataset_name = t[2]
        obstacles_world = load_world_obstacle_polygons(data_paths,dataset_name)
    else:
        obstacles_world = None

    # Trayectory coordinates
    path_file = os.path.join(directory, 'mundo/mun_pos.csv')

    print(path_file)
    raw_traj_data = np.genfromtxt(path_file, delimiter=',')

    # We suppose that the frame ids are in ascending order
    frame_ids = np.unique(raw_traj_data[:, 0]).tolist()
    print("[INF] Total number of frames: ",len(frame_ids))

    raw_traj_data_per_frame = [] # people in frame
    # Group the spatial pedestrian data frame by frame
    # id_frame, id_person, x, y
    for frame in frame_ids:
        raw_traj_data_per_frame.append(raw_traj_data[raw_traj_data[:, 0]==frame, :])
    # Iterate over the frames
    for idx, frame in enumerate(frame_ids):
        # Frame sequence of size seq_len = obs+pred starting at frame
        # id_frame, id_person, x, y por every person present in the frame
        raw_seq_data = raw_traj_data_per_frame[idx:idx+seq_len]
        if args.intersection:
            # Intersection of the id_person of "raw_seq_data"
            peds_id_list = reduce(set.intersection,
                                [set(peds_id_list[:,1]) for peds_id_list in
                                raw_seq_data])

            peds_id_list = sorted(list(peds_id_list))


            raw_seq_data = np.concatenate(raw_seq_data,axis=0)

            # Number of people from the intersection of the id_person of "raw_seq_data"
            num_ped_in_frame = len(peds_id_list)
        else:
            raw_seq_data = np.concatenate(raw_seq_data,axis = 0)
            # Unique indices for the persons in the sequence "raw_seq_data"
            peds_in_cur_seq = np.unique(raw_seq_data[:,1])
            # List of all the persons in this sequence
            peds_id_list    = list(peds_in_cur_seq)
            # Number of unique persons "raw_seq_data"
            num_ped_in_frame = len(peds_in_cur_seq)

        # The following two arrays have the same shape
        # "pos_seq_data" contains all the absolute positions of all the pedestrians in the sequence
        # and he information is encoded in an absolute frame (no transformation)
        pos_seq_data     = np.zeros((num_ped_in_frame, seq_len, 2), dtype="float32")
        # Same, with only the displacements
        rel_seq_data  = np.zeros((num_ped_in_frame, seq_len, 2), dtype="float32")
        # Is the array that have the sequence of Id_person of all people that there are in frame sequence
        frame_ids_seq_data = np.zeros((num_ped_in_frame, seq_len), dtype="int32")
        # When using "keypoints" information
        if args.add_kp:
            # Pixel coordinates, in image absolute coordinates
            kp_feat      = np.zeros((num_ped_in_frame, seq_len, args.kp_num, 3),dtype="float32")
            # Pixel coordinates, in image relative coordinates
            kp_feat_rel = np.zeros((num_ped_in_frame,  seq_len, args.kp_num, 3),dtype="float32")

        # When using social context information
        if args.add_social:
            #Maximum number of persons in a frame
            person_max  = args.person_max
            # absolute pixel-based data: id_person, x, y
            neighbors_data  = np.zeros((num_ped_in_frame, seq_len, person_max, 3),dtype="float32")

        count_ped = 0
        # For all the persons appearing in this sequence that starts at frame
        # We will make one entry in the sequences list
        for ped_id in peds_id_list:
            # Get the information about ped_id, in the whole sequence
            ped_seq_data = raw_seq_data[raw_seq_data[:, 1] == ped_id, :]
            # We want pedestrians whose number of observations inside this sequence is exactly seq_len
            if len(ped_seq_data) != seq_len:
                # We do not have enough observations for this person
                continue
            # -----------------------------------------------------------------------
            # AQUI SE VERIFICA QUE LAS PRIMERAS 8 POSICIONES NO SEAN IGUALES
            # -----------------------------------------------------------------------
            #if(num_ped_in_frame>max_lon):
            #    max_lon=num_ped_in_frame

            # Social context information is extracted here
            # List of all the persons in the frame, to build the neighbors array
            if args.add_social:
                # Check whether the first 8 positions are not the same
                con_iguales=0
                for n in range(obs_len-1):
                    if((ped_seq_data[n,2]==ped_seq_data[n+1,2]) and (ped_seq_data[n,3]==ped_seq_data[n+1,3])):
                        con_iguales +=1
                if (con_iguales==obs_len-1):
                    continue

                # To keep neighbors data for the person ped_id
                neighbors_ped_seq = np.zeros((seq_len, person_max, 3),dtype="float32")

                # Scan all the frames of the sequence
                for frame_idx,frame_id in enumerate(np.unique(raw_seq_data[:,0]).tolist()):
                    # Information of frame "frame_id"
                    frame_data = raw_seq_data[raw_seq_data[:,0] == frame_id, :]
                    # Id, x, y of the pedestrians of frame "num_frame"
                    frame_data = frame_data[:,1:4]
                    # For all the persons in the sequence
                    for neighbor_ped_idx, neighbor_ped_id  in enumerate(peds_id_list):
                        # Get the data of this specific person
                        sped = frame_data[frame_data[:, 0] == neighbor_ped_id , :]
                        # If we have information for this pedestrian, add it to the neighbors struture
                        if sped.size != 0:
                            neighbors_ped_seq[frame_idx,neighbor_ped_idx,:] = sped
                # Contains the neighbor data for count_ped
                neighbors_data[count_ped,:,:,:] = neighbors_ped_seq
            # Spatial data (absolute)
            ped_seq_pos = ped_seq_data[:, 2:]
            # Spatial data (relative)
            ped_seq_rel = np.zeros_like(ped_seq_pos)
            # First frame of the relative array is set to zeros
            ped_seq_rel[1:, :] = ped_seq_pos[1:, :] - ped_seq_pos[:-1, :]
            # Absolute x,y and displacements for all ped_id
            pos_seq_data[count_ped, :, :] = ped_seq_pos
            rel_seq_data[count_ped, :, :] = ped_seq_rel

            # List of frames of any ped_id in this sequence
            frame_ids_seq = frame_ids[idx:idx+seq_len]

            # For each tracked person
            # we keep the list of all the frames in which it is present
            frame_ids_seq_data[count_ped, :] = frame_ids_seq

            # List of ped_ids that had a sequence (may be repeated)
            key_idx.append(ped_id)

            # Keypoints
            if args.add_kp:
                # get the kp feature from starting frame to seq_len frame)
                # key_idx.append(ped_id)
                for i, frame_id in enumerate(frame_ids_seq):
                    #print(frame_id)
                    key = "%d_%d" % (frame_id, ped_id)
                    kp_feat[count_ped, i, :, :] = kp_feats[key][:, :3]

                # puse un 1 por que al inicio no se a movido asi que si estamos seguro de
                # que el desplazamiento del peaton sea cero en todos sus keypoints
                #kp_feat_rel[count_perd, 0, :, 3]= 1.0

                kp_feat_rel[count_ped, 1:, :, :2] = kp_feat[count_ped, 1:, :, :2] - kp_feat[count_ped, :-1, :, :2]
                kp_feat_rel[count_ped, 1:, :,  2] = kp_feat[count_ped, 1:, :,  2] * kp_feat[count_ped, :-1, :, 2]
                kp_feat_rel[count_ped,  0, :,  2] = np.ones((18,))
            # Increment count_person
            count_ped += 1
        # Number of persons getting a sequence starting at this frame
        num_person_in_start_frame.append(count_ped)

        # only count_ped data is preserved
        seq_list_pos.append(pos_seq_data[:count_ped])
        seq_list_rel.append(rel_seq_data[:count_ped])
        seq_list_frames.append(frame_ids_seq_data[:count_ped])

        # Keypoints
        if args.add_kp:
            kp_list.append(kp_feat[:count_ped])
            kp_list_rel.append(kp_feat_rel[:count_ped])
        # Social interactions
        if args.add_social:
            seq_list_person.append(neighbors_data [:count_ped])
            #seq_list_person_rel.append(neighbors_data _rel[:count_ped])
        #hasta aqui
    #print(" maximo numero de personas que permanecen en toda una secuencia de frames")
    #print(max_p)
    #print("max_person long")
    #print(max_lon)

    # Concatenate all the content of the lists (pos/relative pos)
    seq_list_pos = np.concatenate(seq_list_pos, axis=0)
    seq_list_rel = np.concatenate(seq_list_rel, axis=0)
    seq_list_frames = np.concatenate(seq_list_frames, axis=0)
    print("[INF] Total number of sample sequences ",len(seq_list_pos))

    # we get the obs traj and pred_traj
    # [total, obs_len, 2]
    # [total, pred_len, 2]
    obs_traj = seq_list_pos[:, :obs_len, :]
    pred_traj = seq_list_pos[:, obs_len:, :]
    frames_obs = seq_list_frames[:,:obs_len]
    obs_traj_rel = seq_list_rel[:, :obs_len, :]
    pred_traj_rel = seq_list_rel[:, obs_len:, :]

    # the starting idx for each frame in the N*K list,
    # [frame_id, 2]
    cum_start_idx = [0] + np.cumsum(num_person_in_start_frame).tolist()
    seq_start_end = np.array([(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])], dtype="int")

    # Save all these data as a dictionary
    data = {
        "obs_traj": obs_traj,
        "obs_traj_rel": obs_traj_rel,
        "pred_traj": pred_traj,
        "pred_traj_rel": pred_traj_rel,
        "key_idx": np.array(key_idx),
        "obs_frames": frames_obs
    }
    # Optional data
    if args.add_social:
        print("[INF] Add social interaction data")
        seq_list_person = np.concatenate(seq_list_person, axis = 0)
        obs_person = seq_list_person[:,:obs_len,:,:]
        pred_person = seq_list_person[:,obs_len:,:,:]
        # Neighbors information
        data.update({
            "obs_neighbors": obs_person,
            "pred_neighbors": pred_person,
        })
        vec = {
            "obs_neighbors": obs_person,
            "key_idx": np.array(key_idx),
            "obs_traj":  obs_traj
        }
        if args.neighborhood:
            of_sim = OpticalFlowSimulator(use_bounds = True, lim=args.lim[args.ind_test])
            flow,vis_neigh,_ = of_sim.compute_opticalflow_batch(vec['obs_neighbors'], vec['key_idx'], vec['obs_traj'],args.obs_len,obstacles_world)
        else:
            if args.obstacles:
                of_sim = OpticalFlowSimulator()
                flow,vis_neigh,vis_obst = of_sim.compute_opticalflow_batch(vec['obs_neighbors'], vec['key_idx'], vec['obs_traj'], args.obs_len,obstacles_world)
            else:
                of_sim = OpticalFlowSimulator()
                flow,vis_neigh,_ = of_sim.compute_opticalflow_batch(vec['obs_neighbors'], vec['key_idx'], vec['obs_traj'],args.obs_len,None)
        all_flow.append(flow)


    if args.add_kp:
        print("[INF] Add appearance data")
        # [N*K, seq_len, 18, 3]
        kp_list     = np.concatenate(kp_list, axis=0)
        kp_list_rel = np.concatenate(kp_list_rel, axis=0)

        obs_kp     = kp_list[:, :obs_len, :, :]
        pred_kp    = kp_list[:, obs_len:, :, :]  # for visualization
        obs_kp_rel = kp_list_rel[:, :obs_len, :, :]

        data.update({
            "obs_kp": obs_kp,
            "obs_kp_rel": obs_kp_rel
        })
    # Optical flow
    if args.add_social:
        all_flow = np.concatenate(all_flow ,axis=0)
        data.update({
            "obs_flow": all_flow
        })
    if args.obstacles:
        data.update({
            "obstacles": obstacles_world
        })
    return data

def process_file_varios(data_dirs, list_max_person, args, delim, lim=[]):
    
    datasets = range(len(list_max_person))
    datasets = list(datasets)
    datasets.remove(args.ind_test)
    list_max_person = np.delete(list_max_person, args.ind_test)

    if(len(lim)!=0):
        lim = np.delete(lim, args.ind_test,axis=0)
        lim = np.reshape(lim,(4,5))

    # All directions of training set
    used_data_dirs = [data_dirs[x] for x in datasets]

    obs_len = args.obs_len
    pred_len = args.pred_len
    seq_len = obs_len + pred_len
    print("[INF] Sequence length (observation+prediction):", seq_len)
    # este conjunto va guardando los id de las personas que si tuvieron secuencias de la longitud deseada
    #key_idx    = []

    num_person_in_start_frame=[]

    seq_list_pos  = []
    seq_list_rel = []
    seq_list_frames = []  # [N, seq_len]

    kp_list   = []  # [N, seq_len, 18, 2]
    kp_list_rel = []

    todo_flujo = []

    for indi,directory in enumerate(used_data_dirs):

        seq_list_person_indi = []
        key_idx_indi         = []
        seq_list_pos_indi        = []
        #se obtiene el nombre del archivo sin importar del punto txt, csv,etc.
        #name_sub_data = os.path.splitext(os.path.basename(sub_data))[0]
        #print(name_sub_data)
        sub_data = os.path.join(directory, 'mundo/mun_pos.csv')
        print(sub_data)

        # Read obstacles
        if args.obstacles:
            print("[INF] Reading obstacle files")
            t = directory.split('/')
            data_paths = t[0]+'/'+t[1]+'/'
            dataset_name = t[2]
            obstacles_world = load_world_obstacle_polygons(data_paths,dataset_name)
        else:
            obstacles_world = None

        #  To use keypoints, we open then from a file
        kp_feats = {} # "frameidx_personId"
        if args.add_kp:
            kp_file_path = os.path.join(directory,'kp_box.csv')
            with open(kp_file_path, "r") as f:

                for line in f:
                    fidxykp = line.strip().split(delim)
                    key = fidxykp[0] + "_" +fidxykp[1]
                    kp_feats[key] = np.array(fidxykp[2:]).reshape(args.kp_num,3)

        # Trayectory coordinates
        raw_traj_data = np.genfromtxt(sub_data, delimiter= delim)

        # We suppose that the frame ids are in ascending order
        frame_ids = np.unique(raw_traj_data[:, 0]).tolist()
        print("[INF] Total number of frames: ",len(frame_ids))


        raw_traj_data_per_frame = [] # people in frame
        # Group the spatial pedestrian data frame by frame
        # id_frame, id_person, x, y
        for frame in frame_ids:
            raw_traj_data_per_frame.append(raw_traj_data[raw_traj_data[:, 0]==frame, :])

        #contador=0
        for idx, frame in enumerate(frame_ids):
            #Frame sequence of size seq_len = obs+pred
            # id_frame, id_person, x, y por every person present in the frame
            raw_seq_data = raw_traj_data_per_frame[idx:idx+seq_len]

            if args.intersection:
                # Intersection of the id_person of "raw_seq_data"
                peds_id_list = reduce(set.intersection,
                                [set(peds_id_list[:,1]) for peds_id_list in
                                raw_seq_data])

                peds_id_list = sorted(list(peds_id_list))


                raw_seq_data = np.concatenate(raw_seq_data,axis=0)

                # Number of people from the intersection of the id_person of "raw_seq_data"
                num_ped_in_frame = len(peds_id_list)
            else:
                raw_seq_data = np.concatenate(raw_seq_data,axis = 0)
                # Unique indices for the persons in the sequence "raw_seq_data"
                peds_in_cur_seq = np.unique(raw_seq_data[:,1])
                # List of all the persons in this sequence
                peds_id_list    = list(peds_in_cur_seq)
                # Number of unique persons "raw_seq_data"
                num_ped_in_frame = len(peds_in_cur_seq)
            # The following two arrays have the same shape
            # "pos_seq_data" contains all the absolute positions of all the pedestrians in the sequence
            # and he information is encoded in an absolute frame (no transformation)
            pos_seq_data = np.zeros((num_ped_in_frame, seq_len, 2), dtype="float32")
            # Same, with only the displacements
            rel_seq_data = np.zeros((num_ped_in_frame, seq_len, 2), dtype="float32")
            # Is the array that have the sequence of Id_person of all people that there are in frame sequence
            frame_ids_seq_data = np.zeros((num_ped_in_frame, seq_len), dtype="int32")
            # When using "keypoints" information
            if args.add_kp:
                # Pixel coordinates, in image absolute coordinates
                kp_feat = np.zeros((num_ped_in_frame, seq_len, args.kp_num, 3),
                        dtype="float32")
                # Pixel coordinates, in image relative coordinates
                kp_feat_rel = np.zeros((num_ped_in_frame, seq_len, args.kp_num, 3),
                            dtype="float32")
            # When using social context information
            if args.add_social:
                #Maximum number of persons in a frame
                person_max = list_max_person[indi]
                # absolute pixel-based data: id_person, x, y
                neighbors_data = np.zeros((num_ped_in_frame, seq_len, person_max, 3),dtype="float32")


            count_ped = 0
            # For all the persons appearing in this sequence that starts at frame
            # We will make one entry in the sequences list
            for ped_id in peds_id_list:
                # Get the information about ped_id, in the whole sequence
                ped_seq_data = raw_seq_data[raw_seq_data[:, 1] == ped_id, :]
                # We want pedestrians whose number of observations inside this sequence is exactly seq_len
                if len(ped_seq_data) != seq_len:
                    # We do not have enough observations for this person
                    continue

                # contador+=1
                # -----------------------------------------------------------------------
                # Social context information is extracted here
                # List of all the persons in the frame, to build the neighbors array
                if args.add_social:
                    con_iguales = 0
                    for n in range(obs_len-1):
                        if((ped_seq_data[n,2]==ped_seq_data[n+1,2]) and (ped_seq_data[n,3]==ped_seq_data[n+1,3])):
                            con_iguales +=1
                    if(con_iguales==obs_len-1):
                        continue

                    # To keep neighbors data for the person ped_id
                    neighbors_ped_seq = np.zeros((seq_len, person_max, 3),dtype="float32")
                    # Scan all the frames of the sequence
                    for frame_idx,frame_id in enumerate(np.unique(raw_seq_data[:,0]).tolist()):
                        # Information of frame "frame_id"
                        sseq_frame_data = raw_seq_data[raw_seq_data[:,0] == frame_id, :]
                        # Id, x, y of the pedestrians of frame "frame_id"
                        sseq_frame_data = sseq_frame_data[:,1:4]
                        # For all the persons in the sequence
                        for neighbor_ped_idx,neighbor_ped_id in enumerate(peds_id_list):
                            # Get the data of this specific person
                            sped = sseq_frame_data[sseq_frame_data[:, 0] == neighbor_ped_id, :]
                            # If we have information for this pedestrian, add it to the neighbors struture
                            if sped.size != 0:
                                neighbors_ped_seq[frame_idx,neighbor_ped_idx,:] = sped
                    # Contains the neighbor data for count_ped
                    neighbors_data[count_ped,:,:,:] = neighbors_ped_seq
                # Spatial data (absolute)
                ped_seq_pos = ped_seq_data[:, 2:]
                # Spatial data (relative)
                ped_seq_rel = np.zeros_like(ped_seq_pos)
                # First frame of the relative array is set to zeros
                ped_seq_rel[1:, :] = ped_seq_pos[1:, :] - ped_seq_pos[:-1, :]
                # Absolute x,y and displacements for all person_id
                pos_seq_data[count_ped, :, :] = ped_seq_pos
                rel_seq_data[count_ped, :, :] = ped_seq_rel

                # la lista de frames que tiene cada ped_id que pertenece a esta secuencia de frames
                frame_ids_seq  = frame_ids[idx:idx+seq_len]

                # For each tracked person
                # we keep the list of all the frames in which it is present
                frame_ids_seq_data[count_ped, :] = frame_ids_seq

                # List of person_ids that had a sequence (may be repeated)
                key_idx_indi.append(ped_id)

                # "Keypoints"
                if args.add_kp:
                    # get the kp feature from starting frame to seq_len frame)
                    # key_idx.append(ped_id)
                    for i, frame_id in enumerate(frame_ids_seq ):
                        key = "%d_%d" % (frame_id, ped_id)
                        # ignore the kp logits
                        kp_feat[count_ped, i, :, :] = kp_feats[key][:, :3]
                    kp_feat_rel[count_ped, 1:, :, :2] = kp_feat[count_ped, 1:, :, :2] - kp_feat[count_ped, :-1, :, :2]
                    kp_feat_rel[count_ped, 1:, :,  2] = kp_feat[count_ped, 1:, :,  2] * kp_feat[count_ped, :-1, :,  2]
                    kp_feat_rel[count_ped, 0, :,   2] = np.ones((18,))
                count_ped += 1

            # Number of persons getting a sequence starting at this frame
            num_person_in_start_frame.append(count_ped)
            # Only count_ped data are preserved in the following three arrays
            seq_list_pos.append(pos_seq_data[:count_ped])
            seq_list_rel.append(rel_seq_data[:count_ped])
            seq_list_frames.append(frame_ids_seq_data[:count_ped])
            # esto es para pasarlo a al momento de calcular el flujo optico individual
            seq_list_pos_indi.append(pos_seq_data[:count_ped])


            # Keypoints
            if args.add_kp:
                kp_list.append(kp_feat[:count_ped])
                kp_list_rel.append(kp_feat_rel[:count_ped])
            # Social interactions
            if args.add_social:
                #seq_list_person.append(neighbors_data[:count_ped])
                seq_list_person_indi.append(neighbors_data[:count_ped])
        #aquiiiiiiiiiiiiiiiiiii
        if args.add_social:
            # la informacion de los vecinos
            seq_list_person_indi = np.concatenate(seq_list_person_indi, axis = 0)

            obs_person = seq_list_person_indi[:,:obs_len,:,:]
            pred_person = seq_list_person_indi[:,obs_len:,:,:]

            seq_list_pos_indi =np.concatenate(seq_list_pos_indi,axis=0)
            print("The numbers of examples is: ", len(seq_list_pos_indi))

            obs_traj = seq_list_pos_indi[:, :obs_len, :]
            pred_traj = seq_list_pos_indi[:, obs_len:, :]

            #se tiene la informacion para calcular el flujo optico
            vec = {
                "obs_person": obs_person,
                "key_idx": np.array(key_idx_indi),
                "obs_traj":  obs_traj
            }

            #print(vec['obs_person'].shape)
            #print(vec['key_idx'].shape)
            #print(vec['obs_traj'].shape)
            if args.neighborhood:
                fo = OpticalFlowSimulator(use_bounds = True, lim=lim[indi,:])
                flujo,vis_neigh,_= fo.compute_opticalflow_batch(vec['obs_person'], vec['key_idx'], vec['obs_traj'],args.obs_len,obstacles_world)
            else:
                if args.obstacles:
                    fo = OpticalFlowSimulator()
                    flujo,vis_neigh,vis_obst = fo.compute_opticalflow_batch(vec['obs_person'], vec['key_idx'], vec['obs_traj'], args.obs_len,obstacles_world)
                else:
                    fo = OpticalFlowSimulator()
                    flujo,vis_neigh,_ = fo.compute_opticalflow_batch(vec['obs_person'], vec['key_idx'], vec['obs_traj'],args.obs_len,
obstacles_world)
            todo_flujo.append(flujo)
            #if(indi==0):
            #    print(flujo)

    # Concatenate all the content of the lists (pos/relative pos)
    seq_list_pos = np.concatenate(seq_list_pos, axis=0)
    seq_list_rel = np.concatenate(seq_list_rel, axis=0)
    # Agregue
    seq_list_frames= np.concatenate(seq_list_frames, axis=0)
    # we get the obs traj and pred_traj
    # [total, obs_len, 2]
    # [total, pred_len, 2]
    obs_traj = seq_list_pos[:, :obs_len, :]
    pred_traj = seq_list_pos[:, obs_len:, :]
    # Agregue
    frame_obs = seq_list_frames[:, :obs_len]
    #frame_pred = seq_list_frames[:, obs_len:]

    obs_traj_rel = seq_list_rel[:, :obs_len, :]
    pred_traj_rel = seq_list_rel[:, obs_len:, :]

    # the starting idx for each frame in the N*K list,
    # [frame_id, 2]
    cum_start_idx = [0] + np.cumsum(num_person_in_start_frame).tolist()
    seq_start_end = np.array([(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])], dtype="int")

    #print(obs_person[0][0])
    # save the data


    data = {
        "obs_traj": obs_traj,
        "obs_traj_rel": obs_traj_rel,
        "pred_traj": pred_traj,
        "pred_traj_rel": pred_traj_rel,
        #"key_idx": np.array(key_idx),
        "frames_obs": frame_obs,
        #"frames_pred": frame_pred,
        #"frame_list":seq_frameidx_list
    }

    #obs_person = np.array(seq_list_person_obs)
    #pred_person = np.array(seq_list_person_pred)


    if args.add_kp:
        # [N*K, seq_len, 17, 2]
        kp_list = np.concatenate(kp_list, axis=0)
        kp_list_rel = np.concatenate(kp_list_rel, axis=0)

        obs_kp = kp_list[:, :obs_len, :, :]
        pred_kp = kp_list[:, obs_len:, :, :]  # for visualization
        obs_kp_rel = kp_list_rel[:, :obs_len, :, :]

        data.update({
            "obs_kp": obs_kp,
            "obs_kp_rel": obs_kp_rel,
            #"pred_kp": pred_kp,
        })
    if args.add_social:
        todo_flujo = np.concatenate(todo_flujo,axis=0)
        #print(todo_flujo.shape)
        data.update({
            "obs_flow": todo_flujo,
        })

    return data
