import os
from tqdm import tqdm
import glob
import numpy as np
from interaction_optical_flow import OpticalFlowSimulator
from interaction_cuadro_optical_flow import OpticalFlowSimulator1
from interaction_optical_flow_obstacles import OpticalFlowSimulator_obstacles
from obstacles import load_world_obstacle_polygons

# En todas estas funciones cuando se usa el modo add_social
# se contempla a los vecinos que pueden no permanecer en toda la secuencia

"""
  Process the data from the given path_file, of only one dataset
  input: The direction of file, arg: obs_len, pred_len,etc. , delim
  return: a  dictionary with obs_traj, obs_traj_rel, obs_person, key_idx, obs_kp_rel
"""
def process_file_modif(path_file, args, delim):
    # Observation length
    obs_len  = args.obs_len
    # Prediction length
    pred_len = args.pred_len
    # Total sequence length
    seq_len = obs_len + pred_len
    print("[INF] Sequence length (observation+prediction): ",seq_len)
    num_person_in_start_frame = []

    seq_list     = []
    seq_list_rel = []
    seq_frames   = []
    kp_list      = []  # [N, seq_len, 18, 3]
    kp_list_rel  = []

    #Estas listas tienen la informacion de frames de todas las sucesiones de tamano seq_len
    seq_list_person     = []
    seq_list_person_rel = []

    #Load other features if necessary
    #Esta lista tendra los Id_person de las personas de cada secucuencia seq_len que se puedan hacer
    key_idx  = []
    kp_feats = {}  # "frameidx_personId"
    # To use keypoints, we open then from a file
    if args.add_kp:
        kp_path = os.path.join(path_file, 'kp_box.csv')
        with open(kp_path, "r") as f:
            for line in f:
                fidxykp = line.strip().split(delim)
                key     = fidxykp[0] + "_" +fidxykp[1]
                #key_idx.append(key)
                kp_feats[key] = np.array(fidxykp[2:]).reshape(args.kp_num,3) 
    # Read obstacles
    if args.obstacles:
        t = path_file.split('/')
        data_paths = t[0]+'/'+t[1]+'/'
        dataset_name = t[2]
        obstacles_world = load_world_obstacle_polygons(data_paths,dataset_name)

    # Trajectory coordinates
    path_file_com = os.path.join(path_file, 'mundo/mun_pos.csv')
    with open(path_file_com, "r") as traj_file:
        for line in traj_file:
            # Format is: id_frame, id_person, x, y
            fidx, pid, x, y = line.strip().split(delim)
            raw_traj_data.append([fidx, pid, x, y])
    # Convert python array to numpy array
    raw_traj_data = np.array(raw_traj_data, dtype="float32")

    # We suppose that the frame ids are in ascending order
    frame_ids               = np.unique(raw_traj_data[:, 0]).tolist()  # Determine the unique frames ids
    raw_traj_data_per_frame = []  # people in frame

    # Group the spatial pedestrian data frame by frame
    # id_frame, id_person, x, y
    for frame in frame_ids:
        raw_traj_data_per_frame.append(raw_traj_data[raw_traj_data[:, 0]==frame, :])

    # Maximum number of persons in a frame sequence
    print("[INF] Total number of tracked ids: ",len(frame_ids))

    # Iterate over the frames
    for idx, frame in enumerate(frame_ids):
        # Frame sequence of size seq_len = obs+pred starting at frame
        # id_frame, id_person, x, y por every person present in the frame
        cur_seq_data = np.concatenate(raw_traj_data_per_frame[idx:idx+seq_len],axis = 0)
        # Unique indices for the persons in the sequence "cur_seq_data"
        persons_in_cur_seq = np.unique(cur_seq_data[:,1])
        # List of all the persons in this sequence
        ped_id_list           = list(persons_in_cur_seq)
        # Number of unique persons "cur_seq_data"
        num_person_in_cur_seq = len(persons_in_cur_seq)

        # The following two arrays have the same shape
        # "cur_seq" contains all the absolute positions of all the pedestrians in the sequence
        # and he information is encoded in an absolute frame (no transformation)
        cur_seq       = np.zeros((num_person_in_cur_seq, seq_len, 2), dtype="float32")
        # Same, with only the displacements
        cur_seq_rel   = np.zeros((num_person_in_cur_seq, seq_len, 2), dtype="float32")
        # In this array, we'll have the sequence of id_person for all the persons in the sequence frames
        # This will be useful to identify neighbors
        cur_seq_frame = np.zeros((num_person_in_cur_seq, seq_len),    dtype="int32")
        # List of frames of any person_id in this sequence
        frame_idxs = frame_ids[idx:idx+seq_len]

        # When using "keypoints" information
        if args.add_kp:
            # Pixel coordinates, in image absolute coordinates
            kp_feat      = np.zeros((num_person_in_cur_seq, seq_len, args.kp_num, 2),dtype="float32")
            # Pixel coordinates, in image relative coordinates
            kp_feat_rel = np.zeros((num_person_in_cur_seq,  seq_len, args.kp_num, 2),dtype="float32")

        # When using social context information
        if args.add_social:
            person_max     = args.person_max
            # absolute pixel-based data: id_person, x, y
            neighbors_data = np.zeros((num_person_in_cur_seq, seq_len, person_max, 3),dtype="float32")

        # This counter is reinitialized every time we move to a new frame
        count_person = 0
        # For all the persons appearing in this sequence that starts at frame
        # We will make one entry in the sequences list
        for person_id in persons_in_cur_seq:
            # Get the information about person_id, in the whole sequence
            cur_person_seq = cur_seq_data[cur_seq_data[:, 1] == person_id,:]
            # We want pedestrians whose number of observations inside this sequence is exactly seq_len
            if len(cur_person_seq) != seq_len:
                # We do not have enough observations for this person
                continue
            # Social context information is extracted here
            # List of all the persons in the frame, to build the neighbors array
            if args.add_social:
                # -----------------------------------------------------------------------
                # Check whether the first 8 positions are not the same
                # TODO: is that only when using add_social?
                # -----------------------------------------------------------------------
                equal_positions = 0
                for n in range(obs_len-1):
                    if((cur_person_seq[n,2]==cur_person_seq[n+1,2]) and (cur_person_seq[n,3]==cur_person_seq[n+1,3])):
                        equal_positions +=1
                if (equal_positions==obs_len-1):
                    continue

                #  To keep neighbors data for the person person_id
                neighbors_person_seq = np.zeros((seq_len, person_max, 3),dtype="float32")

                seq_frame = 0
                # Scan all the frames of the sequence
                for num_frame in np.unique(cur_seq_data[:,0]).tolist():
                    # Information of frame "num_frame"
                    sseq_frame_data = cur_seq_data[cur_seq_data[:,0] == num_frame,:]
                    # Id, x, y of the pedestrians of frame "num_frame"
                    sseq_frame_data = sseq_frame_data[:,1:4]
                    # For all the persons in the sequence
                    for ped in range(num_person_in_cur_seq):
                        # Get the person Id
                        ped_id = ped_id_list[ped]
                        # Get the data of this specific person
                        sped = sseq_frame_data[sseq_frame_data[:, 0] == ped_id, :]
                        # If we have information for this pedestrian, add it to the neighbors struture
                        if sped.size != 0:
                            neighbors_person_seq[seq_frame,ped,:] = sped
                    # Increment the array index
                    seq_frame+=1
                #
                neighbors_data[count_person,:,:,:] = neighbors_person_seq

            # Spatial data (absolute)
            cur_person_seq     = cur_person_seq[:, 2:]
            # Spatial data (relative)
            cur_person_seq_rel = np.zeros_like(cur_person_seq)
            # First frame of the relative array is set to zeros
            cur_person_seq_rel[1:, :] = cur_person_seq[1:, :] - cur_person_seq[:-1, :]
            # Absolute x,y and displacements for all person_id
            cur_seq[count_person, :, :]     = cur_person_seq
            cur_seq_rel[count_person, :, :] = cur_person_seq_rel

            # For each tracked person kept in cur_seq_frame
            # we keep the list of all the persons present in this sequence
            cur_seq_frame[count_person, :] = frame_idxs

            # List of person_ids that had a sequence
            key_idx.append(person_id)

            # Keypoints
            if args.add_kp:
                # get the kp feature from starting frame to seq_len frame)
                # key_idx.append(person_id)
                for i, frame_idx in enumerate(frame_idxs):
                    key = "%d_%d" % (frame_idx, person_id)
                    # ignore the kp logits
                    kp_feat[count_person, i, :, :] = kp_feats[key][:, :3]

                kp_feat_rel[count_person, 1:, :, :2] = kp_feat[count_person, 1:, :, :2] - kp_feat[count_person, :-1, :, :2]
                kp_feat_rel[count_person, 1:, :,  2] = kp_feat[count_person, 1:, :,  2] * kp_feat[count_person, :-1, :,  2]
                kp_feat_rel[count_person, 0,  :,  2] = np.ones((18,))
            count_person += 1

        # Number of persons getting a sequence starting at this frame
        num_person_in_start_frame.append(count_person)
        # only count_person data are preserved
        seq_list.append(cur_seq[:count_person])
        seq_list_rel.append(cur_seq_rel[:count_person])
        seq_frames.append(cur_seq_frame[:count_person])


        # Keypoints
        if args.add_kp:
            kp_list.append(kp_feat[:count_person])
            kp_list_rel.append(kp_feat_rel[:count_person])
        # Social interactions
        if args.add_social:
            seq_list_person.append(neighbors_data[:count_person])

    # N is numero de secuencias de frames  for each video, K is num_person in each frame
    # el numero total que tendremos es el numero total de personas que hayan cumplido que si tienen secuencia
    seq_list     = np.concatenate(seq_list, axis=0)
    seq_list_rel = np.concatenate(seq_list_rel, axis=0)
    seq_frames   = np.concatenate(seq_frames, axis=0)

    print("[INF] Total number of examples ",len(seq_list))

    # we get the obs traj and pred_traj
    # [total, obs_len, 2]
    # [total, pred_len, 2]
    obs_traj = seq_list[:, :obs_len, :]
    pred_traj = seq_list[:, obs_len:, :]
    frames_obs = seq_frames[:,:obs_len]

    obs_traj_rel = seq_list_rel[:, :obs_len, :]
    pred_traj_rel = seq_list_rel[:, obs_len:, :]


    # the starting idx for each frame in the N*K list,
    # [num_frame, 2]
    cum_start_idx = [0] + np.cumsum(num_person_in_start_frame).tolist()
    seq_start_end = np.array([
        (start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])
      ], dtype="int")

    # save the data as a dictionary
    data = {
        "obs_traj": obs_traj,
        "obs_traj_rel": obs_traj_rel,
        "pred_traj": pred_traj,
        "pred_traj_rel": pred_traj_rel,
        "key_idx": np.array(key_idx),
        "obs_frames": frames_obs
    }

    #obs_person = np.array(seq_list_person_obs)
    #pred_person = np.array(seq_list_person_pred)
    if args.add_social:
        seq_list_person = np.concatenate(seq_list_person, axis = 0)
        #seq_list_person_rel = np.concatenate(seq_list_person_rel, axis = 0)

        #pred_person = np.array(seq_list_person_pred)
        obs_person = seq_list_person[:,:obs_len,:,:]
        pred_person = seq_list_person[:,obs_len:,:,:]
        #obs_person_rel = seq_list_person_rel[:,:obs_len,:,:]
        #pred_person_rel = seq_list_person_rel[:,obs_len:,:,:]

        data.update({
            "obs_person": obs_person,
            "pred_person": pred_person,

            #"obs_person_rel": obs_person_rel,
            #"pred_person_rel": pred_person_rel,
        })

    if args.add_kp:
        # [N*K, seq_len, 18, 3]
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
    if args.obstacles:
        data.update({
            "obstacles:" obstacles_world,
        })

    return data

"""
  Process the data from several file, the that form the set of training and evaluation
  and the file of test is process with process_file_modif
  input: a array whit the directions the all file, a array with max_person of each file
   of the array of direcctions,  arg: obs_len, pred_len, ind_test, delim
  return: a  dictionary with obs_traj, obs_traj_rel, obs_person, key_idx, obs_kp_rel, obs_flujo
"""
def process_file_modif_varios(data_dirs, list_max_person, args, delim, lim =[]):

    datasets = range(len(list_max_person))
    datasets = list(datasets)
    datasets.remove(args.ind_test)
    list_max_person = np.delete(list_max_person, args.ind_test) 

    if(len(lim)!=0):
        lim = np.delete(lim, args.ind_test,axis=0)
        lim = np.reshape(lim,(4,5))

    #Las direcciones del conjunto de entrenamiento
    used_data_dirs = [data_dirs[x] for x in datasets]

    obs_len = args.obs_len
    pred_len = args.pred_len
    seq_len = obs_len + pred_len

    # va guardando los frames de cada secuencia buena
    seq_frames = []  # [N, seq_len]

    num_person_in_start_frame=[]

    seq_list     = []
    seq_list_rel = []

    kp_list   = []  # [N, seq_len, 18, 2]
    kp_list_rel = []

    todo_flujo =[]

    for indi,directory in enumerate(used_data_dirs):

        seq_list_person_indi = []
        key_idx_indi         = []
        seq_list_indi        = []

        sub_data= os.path.join(directory, 'mundo/mun_pos.csv')
        print(sub_data)

        kp_feats = {} # "frameidx_personId"
        if args.add_kp:
            kp_file_path= os.path.join(directory,'kp_box.csv')
            with open(kp_file_path, "r") as f:

                for line in f:
                    fidxykp = line.strip().split(delim)
                    key = fidxykp[0] + "_" +fidxykp[1]
                    kp_feats[key] = np.array(fidxykp[2:]).reshape(args.kp_num,3)
        # Read obstacles
        if args.obstacles:
            t = directory.split('/')
            data_paths = t[0]+'/'+t[1]+'/'
            dataset_name = t[2]
            obstacles_world = load_world_obstacle_polygons(data_paths,dataset_name)

        # Trajectory coordinates
        data = []
        with open(sub_data, "r") as traj_file:
            for line in traj_file:
                #id_frame, id_person, x, y
                fidx, pid, x, y = line.strip().split(delim)
                data.append([fidx, pid, x, y])
        # Convert python array to numpy array
        data = np.array(data, dtype="float32")

        # We suppose that the frame ids are in ascending order
        frames     = np.unique(data[:, 0]).tolist()  # id_frames
        frame_data = []  # people in frame

        # Group the pedstrian data frame by frame
        # id_frame, id_person, x, y
        for frame in frames:
            frame_data.append(data[data[:, 0]==frame, :])

        #contador=0

        for idx, frame in enumerate(frames):
            #la secuencia de frames de size seq_len=obs+pred
            cur_seq_data = np.concatenate(frame_data[idx:idx + seq_len],axis=0)

            # Unique indices for the persons in the sequence "cur_seq_data"
            persons_in_cur_seq = np.unique(cur_seq_data[:,1])

            # El numero de personas de la interseccion de los id_person de cur_seq_data
            num_person_in_cur_seq = len(persons_in_cur_seq)

            # los siguientes dos array tienen la misma forma
            # tiene toda la informacion de todas las personas que hay en la secuencia de frames
            # y la informacion que habra sera de x,y de manera absoluta
            # de forma absoluta (sin ninguna transformacion)

            cur_seq = np.zeros((num_person_in_cur_seq, seq_len, 2), dtype="float32")
            # Este array llevara toda la informacion de todas las personas que hay en esta secuencia
            # de frames, y lo que habra sera sus desplazamientos
            # por ejemplo la primera posicion del la secuencia de frames todostienen desplazamiento cero
            # por que no han avanzado

            cur_seq_rel = np.zeros((num_person_in_cur_seq, seq_len, 2), dtype="float32")
            # Es el array que tienen la secuencia de Id de todas las personas que hay
            # en las secuencia de frames
            cur_seq_frame = np.zeros((num_person_in_cur_seq, seq_len), dtype="int32")


            if args.add_kp:
                # absolute pixel
                kp_feat = np.zeros((num_person_in_cur_seq, seq_len, args.kp_num, 3),
                        dtype="float32")
                # velocity
                kp_feat_rel = np.zeros((num_person_in_cur_seq, seq_len, args.kp_num, 3),
                            dtype="float32")

            if args.add_social:
                person_max     = list_max_person[indi]
                # Absolute pixel
                neighbors_data = np.zeros((num_person_in_cur_seq, seq_len, person_max, 3),dtype="float32")

            # se inicializa cada que cambiamos de secuencia de frames
            count_person = 0
            # Aqui se recorre cada una de las personas que hay en toda la secuencia de frames
            for person_id in persons_in_cur_seq:
                # se obtiene toda la informacion de persona person_id presente en la secuencia de frames
                cur_person_seq = cur_seq_data[cur_seq_data[:, 1] == person_id, :]

                # este if es para asegurar que todas las personas tengan secuencias de longitud seq_len

                if len(cur_person_seq) != seq_len:
                    # se omite la secuencia que no cubre todos loa frames
                    continue

                # -----------------------------------------------------------------------
                # Creamos lista con las personas de cada frame
                # esto es con el fin de tener la informacion de todos sus vecinos de la persona_id actual
                # esto se inicializa cada que cambiamos de una sucesion de seq_len
                if args.add_social:

                    con_iguales=0
                    for n in range(obs_len-1):
                        if((cur_person_seq[n,2]==cur_person_seq[n+1,2]) and (cur_person_seq[n,3]==cur_person_seq[n+1,3])):
                            con_iguales +=1
                    if(con_iguales==obs_len-1):
                        continue

                    #cur_frame_seq = []
                    #Recorremos cada uno de los frames en los que esta la person_id actual
                    pedID_list = list(persons_in_cur_seq)

                    # recorremos cada uno de los frames de la secuencia
                    vecinos_person_seq = np.zeros((seq_len, person_max, 3),dtype="float32")

                    seq = 0
                    for num_frame in np.unique(cur_seq_data[:,0]).tolist():
                        # Toda la informacion del frame num_frame
                        sseq_frame_data = cur_seq_data[cur_seq_data[:,0] == num_frame, :]
                        # solo se toman los Id,x,y de toda  informacion
                        sseq_frame_data = sseq_frame_data[:,1:4]
                        for ped in range(num_person_in_cur_seq):
                            pedID = pedID_list[ped]
                            sped = sseq_frame_data[sseq_frame_data[:, 0] == pedID, :]
                            if sped.size != 0:
                                #sourceData[count_person,seq, ped, :] = sped
                                vecinos_person_seq[seq,ped,:] = sped
                        seq+=1
                    neighbors_data[count_person,:,:,:] = vecinos_person_seq
                # [seq_len,2], solo tiene x,y
                cur_person_seq = cur_person_seq[:, 2:]
                cur_person_seq_rel = np.zeros_like(cur_person_seq)

                # first frame is zeros x,y
                cur_person_seq_rel[1:, :] = cur_person_seq[1:, :] - cur_person_seq[:-1, :]
                # es la informacion de x,y y los desplazamientos de todas las person_id que hubieron
                # en cada secuencia de frames

                cur_seq[count_person, :, :] = cur_person_seq
                cur_seq_rel[count_person, :, :] = cur_person_seq_rel

                # la lista de frames que tiene cada person_id que pertenece a esta secuencia de frames
                frame_idxs = frames[idx:idx+seq_len]

                # Por cada person_id en la secuencia de frames guardamos la secuencia de frames
                #obviamente todas las personas que esten en una misma secuencia de frames, van a tener los
                # la misma lista de frames
                cur_seq_frame[count_person, :] = frame_idxs

                # tiene el person_id de cada una de las personas que si tuvieron secuencias de longitud seq_len
                key_idx_indi.append(person_id)

                #Si agregamos informacion de pose  "keypoints"
                if args.add_kp:
                    # get the kp feature from starting frame to seq_len frame)
                    # key_idx.append(person_id)
                    for i, frame_idx in enumerate(frame_idxs):
                        key = "%d_%d" % (frame_idx, person_id)
                        # ignore the kp logits
                        kp_feat[count_person, i, :, :] = kp_feats[key][:, :3]
                    kp_feat_rel[count_person, 1:, :, :2] = kp_feat[count_person, 1:, :, :2] - kp_feat[count_person, :-1, :, :2]
                    kp_feat_rel[count_person, 1:, :,  2] = kp_feat[count_person, 1:, :,  2] * kp_feat[count_person, :-1, :,  2]
                    kp_feat_rel[count_person, 0,  :,  2] = np.ones((18,))
                count_person += 1

            # Es el vector de que cuenta por cada sucesion de frames cuantas personas por cada sucesion si
            # tuvieron la longitud deseada
            num_person_in_start_frame.append(count_person)

            # Solo las personas "count_person" se preserva su informacion

            seq_list.append(cur_seq[:count_person])
            seq_list_rel.append(cur_seq_rel[:count_person])
            seq_frames.append(cur_seq_frame[:count_person])
            # esto es para pasarlo a al momento de calcular el flujo optico individual
            seq_list_indi.append(cur_seq[:count_person])


            # Other characteristics
            if args.add_kp:
                kp_list.append(kp_feat[:count_person])
                kp_list_rel.append(kp_feat_rel[:count_person])
            if args.add_social:
                seq_list_person_indi.append(neighbors_data[:count_person])
        #aquiiiiiiiiiiiiiiiiiii
        if args.add_social:
            # la informacion de los vecinos
            seq_list_person_indi = np.concatenate(seq_list_person_indi, axis = 0)

            obs_person = seq_list_person_indi[:,:obs_len,:,:]
            pred_person = seq_list_person_indi[:,obs_len:,:,:]

            seq_list_indi =np.concatenate(seq_list_indi,axis=0)

            obs_traj = seq_list_indi[:, :obs_len, :]
            pred_traj = seq_list_indi[:, obs_len:, :]

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
                fo = OpticalFlowSimulator1()
                flujo,vis_neigh = fo.compute_opticalflow_batch_with_neighborhood(vec['obs_person'], vec['key_idx'], vec['obs_traj'],args.obs_len,lim[indi,:])
            else:
                if args.obstacles:
                    fo = OpticalFlowSimulator_obstacles()
                    flujo,vis_neigh,vis_obst = fo.compute_opticalflow_batch(vec['obs_person'], vec['key_idx'], vec['obs_traj'], args.obs_len,obstacles_world)  
                else:
                    fo = OpticalFlowSimulator()
                    flujo,vis_neigh = fo.compute_opticalflow_batch(vec['obs_person'], vec['key_idx'], vec['obs_traj'],args.obs_len)
            todo_flujo.append(flujo)
          
    # N is numero de secuencias de frames  for each video, K is num_person in each frame
    # el numero total que tendremos es el numero total de personas que hayan cumplido que si tienen secuencia
    seq_list = np.concatenate(seq_list, axis=0)
    seq_list_rel = np.concatenate(seq_list_rel, axis=0)
    # Agregue
    seq_frames= np.concatenate(seq_frames, axis=0)
    # we get the obs traj and pred_traj
    # [total, obs_len, 2]
    # [total, pred_len, 2]
    obs_traj = seq_list[:, :obs_len, :]
    pred_traj = seq_list[:, obs_len:, :]
    # Agregue
    frame_obs = seq_frames[:, :obs_len]
    #frame_pred = seq_frames[:, obs_len:]

    obs_traj_rel = seq_list_rel[:, :obs_len, :]
    pred_traj_rel = seq_list_rel[:, obs_len:, :]

    # the starting idx for each frame in the N*K list,
    # [num_frame, 2]
    cum_start_idx = [0] + np.cumsum(num_person_in_start_frame).tolist()
    seq_start_end = np.array([(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])], dtype="int")

    data = {
        "obs_traj": obs_traj,
        "obs_traj_rel": obs_traj_rel,
        "pred_traj": pred_traj,
        "pred_traj_rel": pred_traj_rel,
        "frames_obs": frame_obs,
        #"frames_pred": frame_pred,
        #"frame_list":seq_frameidx_list
    }

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
        })
    if args.add_social:
        todo_flujo = np.concatenate(todo_flujo,axis=0)
        print(todo_flujo.shape)
        data.update({
            "obs_flujo": todo_flujo,
        })

    return data




"""
  Process the data from several file, the that form the set of training and evaluation
  if  your respective obs_flujo was already computed.
  input: a array with the indices of the file that they will be processed,  arg: obs_len, pred_len,etc.
         ,delim
  return: a  dictionary with obs_traj, obs_traj_rel, obs_person, obs_kp_rel, obs_flujo
"""
def datos_subprocesados_varios(datasets, args, delim):

    data_dirs = ['../data1/eth-univ/mundo', '../data1/eth-hotel/mundo',
                 '../data1/ucy-zara01/mundo', '../data1/ucy-zara02/mundo',
                 '../data1/ucy-univ/mundo']

    name_flujo = ['eth-univ','eth-hotel','ucy-zara01','ucy-zara02','ucy-univ']

    used_data_dirs = [data_dirs[x] for x in datasets]

    used_name = [name_flujo[x] for x in datasets]

    obs_len = args.obs_len
    pred_len = args.pred_len
    seq_len = obs_len + pred_len

    # este conjunto va guardando los id de las personas que si tuvieron secuencias de la longitud deseada
    key_idx=[]

    # va guardando los frames de cada secuencia buena
    seq_frameidx_list = []  # [N, seq_len]

    num_person_in_start_frame=[]


    seq_list=[]
    seq_list_rel=[]

    kp_list = []  # [N, seq_len, 18, 2]
    kp_list_rel = []

    # aqui tengo todas las direcciones de los sub_datas
    #sub_datas = glob.glob(os.path.join(path_file,"*.csv"))
    #print(sub_datas)

    for indi,directory in enumerate(used_data_dirs):

        name_sub_flujo ='flujo_bien_non_'+ used_name[indi]+'_total_mundo_12.npy'

        directory_sub_flujo = os.path.join(args.directory_flujo,name_sub_flujo)
        # se carga el archivo del flujo
        sub_flujo = np.load(directory_sub_flujo)
        if indi==0:
            todo_flujo = sub_flujo
        if indi>0:
            todo_flujo = np.concatenate((todo_flujo,sub_flujo),axis=0)

        sub_data= os.path.join(directory, 'mun_pos.csv')
        print(sub_data)

        kp_feats = {} # "frameidx_personId"
        if args.add_kp:
            kp_file_path= os.path.join(directory,'kp_box.csv')

            with open(kp_file_path, "r") as f:

                for line in f:
                    fidxykp = line.strip().split(delim)
                    key = fidxykp[0] + "_" +fidxykp[1]
                    kp_feats[key] = np.array(fidxykp[2:]).reshape(args.kp_num,3)
        data=[]
        with open(sub_data, "r") as traj_file:
            for line in traj_file:
                #fidx, pid, x, y = line.strip().split('\t')
                fidx, pid, x, y = line.strip().split(delim)
                data.append([fidx, pid, x, y])
        data = np.array(data, dtype="float32")

        frames = np.unique(data[:, 0]).tolist()  # all frame_idx
        frame_data = []

        #toda la informacion de cada frame
        for frame in frames:
            frame_data.append(data[frame == data[:, 0], :])

        #contador = 0
        for idx, frame in enumerate(frames):
            #la secuencia de frames de size seq_len=obs+pred
            cur_seq_data = np.concatenate(frame_data[idx:idx + seq_len], axis = 0)

            # los indices unicos de las personas en toda la secuencoa de frames
            persons_in_cur_seq = np.unique(cur_seq_data[:, 1])

            # El numero de personas unicas que hay en la secuencia de frames
            num_person_in_cur_seq = len(persons_in_cur_seq)

            # los siguientes dos array tienen la misma forma
            # tiene toda la informacion de todas las personas que hay en la secuencia de frames
            # y la informacion que habra sera de x,y de manera absoluta
            # de forma absoluta (sin ninguna transformacion)

            cur_seq = np.zeros((num_person_in_cur_seq, seq_len, 2), dtype="float32")
            # Este array llevara toda la informacion de todas las personas que hay en esta secuencia
            # de frames, y lo que habra sera sus desplazamientos
            # por ejemplo la primera posicion del la secuencia de frames todostienen desplazamiento cero
            # por que no han avanzado

            cur_seq_rel = np.zeros((num_person_in_cur_seq, seq_len, 2), dtype="float32")
            # Es el array que tienen la secuencia de Id de todas las personas que hay
            # en las secuencia de frames
            cur_seq_frame = np.zeros((num_person_in_cur_seq, seq_len), dtype="int32")


            if args.add_kp:
                # absolute pixel
                kp_feat = np.zeros((num_person_in_cur_seq, seq_len, args.kp_num, 2),
                        dtype="float32")
                # velocity
                kp_feat_rel = np.zeros((num_person_in_cur_seq, seq_len, args.kp_num, 2),
                            dtype="float32")

            # se inicializa cada que cambiamos de secuencia de frames
            count_person = 0

            # Aqui se recorre cada una de las personas que hay en toda la secuencia de frames
            for person_id in persons_in_cur_seq:
                # se obtiene toda la informacion de persona person_id presente en la secuencia de frames
                cur_person_seq = cur_seq_data[cur_seq_data[:, 1] == person_id, :]

                # este if es para asegurar que todas las personas tengan secuencias de longitud seq_len
                if len(cur_person_seq) != seq_len:
                    # se omite la secuencia que no cubre todos loa frames
                    continue

                if args.add_social:
                    con_iguales=0
                    for n in range(obs_len-1):
                        if((cur_person_seq[n,2]==cur_person_seq[n+1,2]) and (cur_person_seq[n,3]==cur_person_seq[n+1,3])):
                            con_iguales +=1
                    if(con_iguales==obs_len-1):
                        continue

                # [seq_len,2], solo tiene x,y
                cur_person_seq = cur_person_seq[:, 2:]
                cur_person_seq_rel = np.zeros_like(cur_person_seq)

                # first frame is zeros x,y
                cur_person_seq_rel[1:, :] = cur_person_seq[1:, :] - cur_person_seq[:-1, :]
                # es la informacion de x,y y los desplazamientos de todas las person_id que hubieron
                # en cada secuencia de frames

                cur_seq[count_person, :, :] = cur_person_seq
                cur_seq_rel[count_person, :, :] = cur_person_seq_rel

                # la lista de frames que tiene cada person_id que pertenece a esta secuencia de frames
                frame_idxs = frames[idx:idx+seq_len]

                # Por cada person_id en la secuencia de frames guardamos la secuencia de frames
                #obviamente todas las personas que esten en una misma secuencia de frames, van a tener los
                # la misma lista de frames
                cur_seq_frame[count_person, :] = frame_idxs

                # tiene el person_id de cada una de las personas que si tuvieron secuencias de longitud seq_len
                key_idx.append(person_id)

                #Si agregamos informacion de pose  "keypoints"
                if args.add_kp:
                    # get the kp feature from starting frame to seq_len frame)
                    # key_idx.append(person_id)
                    for i, frame_idx in enumerate(frame_idxs):
                        key = "%d_%d" % (frame_idx, person_id)
                        # ignore the kp logits
                        kp_feat[count_person, i, :, :] = kp_feats[key][:, :3]
                    kp_feat_rel[count_person, 1:, :, :2] = kp_feat[count_person, 1:, :, :2] - kp_feat[count_person, :-1, :, :2]
                    kp_feat_rel[count_person, 1:, :,  2] = kp_feat[count_person, 1:, :,  2] * kp_feat[count_person, :-1, :,  2]
                    kp_feat_rel[count_person, 0,  :,  2] = np.ones((18,))
                count_person += 1

            # contador cuenta cuanta subsucesiones hubier+on por cada sud_archivo
            #print("El numero de secuencias")
            #print(contador)
            # Es el vector de que cuenta por cada sucesion de frames cuantas personas por cada sucesion si
            # tuvieron la longitud deseada
            num_person_in_start_frame.append(count_person)

            # Solo las personas "count_person" se preserva su informacion
            seq_list.append(cur_seq[:count_person])
            seq_list_rel.append(cur_seq_rel[:count_person])
            seq_frameidx_list.append(cur_seq_frame[:count_person])

            #Otras caracteristicas
            if args.add_kp:
                kp_list.append(kp_feat[:count_person])
                kp_list_rel.append(kp_feat_rel[:count_person])
    num_seq = len(seq_list)  # el numero total se secuencias de frames de tamano seq_len que hubieron,sea

    print(todo_flujo.shape)

    #seq_frameidx_list=np.concatenate(seq_frameidx_list,axis=0)
    # N is numero de secuencias de frames  for each video, K is num_person in each frame
    # el numero total que tendremos es el numero total de personas que hayan cumplido que si tienen secuencia
    seq_list = np.concatenate(seq_list, axis=0)
    seq_list_rel = np.concatenate(seq_list_rel, axis=0)
    # Agregue
    seq_frameidx_list = np.concatenate(seq_frameidx_list, axis=0)
    # we get the obs traj and pred_traj
    # [total, obs_len, 2]
    # [total, pred_len, 2]
    obs_traj = seq_list[:, :obs_len, :]
    pred_traj = seq_list[:, obs_len:, :]
    # Agregue
    frame_obs = seq_frameidx_list[:, :obs_len]
    frame_pred = seq_frameidx_list[:, obs_len:]

    obs_traj_rel = seq_list_rel[:, :obs_len, :]
    pred_traj_rel = seq_list_rel[:, obs_len:, :]

    # the starting idx for each frame in the N*K list,
    # [num_frame, 2]
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
        #"frames_obs": frame_obs,
        #"frames_pred": frame_pred,
    }

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
    data.update({
    "obs_flujo": todo_flujo,
        })

    return data
