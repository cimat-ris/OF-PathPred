import os
from tqdm import tqdm
import glob
import numpy as np
from interaction_optical_flow import OpticalFlowSimulator 
from interaction_cuadro_optical_flow import OpticalFlowSimulator1
from interaction_optical_flow_obstacles import OpticalFlowSimulator_obstacles
from obstacles import load_world_obstacle_polygons

# En todas estas  cuando se usa el modo add_social solo se toma en cuenta a los vecinos 
# que permanecen en toda la secuencia


# Funcion para preprocesar los datos cargados.
# Esta funcion solo procesa los datos de un conjunto de datos que estan en un solo archivo
def process_file_modif_inter(directory, args, delim):
    
    
    obs_len = args.obs_len
    pred_len = args.pred_len
    seq_len = obs_len + pred_len
    
    print("[INF] Sequence length: ", seq_len)
    num_person_in_start_frame = []
    
    seq_list     = []
    seq_list_rel = []
    seq_frames   = []
    kp_list      = []  # [N, seq_len, 17, 2]
    kp_list_rel  = []
    
    #Estas listas tienen la informacion de frames de todas las sucesiones de tamano seq_len
    seq_list_person     = []
    seq_list_person_rel = []
    
    # Load other features if necessary
    # Esta lista tendra los Id_person de las personas de cada secuencia seq_len que se puedan hacer
    key_idx = []
    kp_feats = {}  # "frameidx_personId"#  To use keypoints, we open then from a file

    if args.add_kp:
        kp_path = os.path.join(directory,'kp_box.csv')
        print(kp_path)
        with open(kp_path, "r") as f:
            for line in f:
                fidxykp = line.strip().split(delim)
                key = fidxykp[0] + "_" +fidxykp[1]
                kp_feats[key] = np.array(fidxykp[2:]).reshape(args.kp_num,3) 
    #obstacles
    if args.obstacles:
        t = directory.split('/')
        data_paths = t[0]+'/'+t[1]+'/'
        dataset_name = t[2]
        obstacles_world = load_world_obstacle_polygons(data_paths,dataset_name)
    
    # Trayectory coordinates
    path_file = os.path.join(directory, 'mundo/mun_pos.csv')
    print(path_file)
    data = np.genfromtxt(path_file, delimiter=',')
    # uniques frames
    frameList = np.unique(data[:, 0]).tolist()
    frame_data = [] # people in frame

    for frame in frameList:
        frame_data.append(data[data[:, 0]==frame, :])

    # Iterate over the frames(frameList)
    #max_p=0
    #max_lon=0
    for idx, frame in enumerate(frameList):
    	#Frame sequence of size seq_len = obs+pred
    	# id_frame, id_person, x, y por every person present in the frame
    	cur_seq_data = frame_data[idx:idx+seq_len]
    	# Intersection of the id_person of "cur_seq_data"
    	nf_ped_ids = reduce(set.intersection,
                                [set(nf_ped_ids[:,1]) for nf_ped_ids in
                                cur_seq_data])
    	
    	nf_ped_ids = sorted(list(nf_ped_ids))

    	cur_seq_data = np.concatenate(cur_seq_data,axis=0)

    	# Number of people from the intersection of the id_person of "cur_Seq_data"   
    	num_ped_in_frame = len(nf_ped_ids)

        #if(num_ped_in_frame>max_p):
        #    max_p = num_ped_in_frame

    	#the next array have the same form
    	#"cur_seq" Contiene toda la informacion de todas las personas que hay en la secuencia de frames 
    	#y la informacion que habra sera de x,y de forma absoluta(sin ninguna transformacion)

    	cur_seq     = np.zeros((num_ped_in_frame, seq_len, 2), dtype="float32")

    	#Este array llevara toda la informacion de todas las personas que hay en esta secuencia 
    	#de frames, y lo que habra sera sus desplazamientos, por ejemplo la primera posicion de la secuencia de frames todos tienen desplazamiento cero
    	#por que no han avanzado
    	cur_seq_rel  = np.zeros((num_ped_in_frame, seq_len, 2), dtype="float32")
    	#Es el array que tienen la secuencia de Id_person de todas las personas que hay en las secuencia de frames
    	cur_seq_frame = np.zeros((num_ped_in_frame, seq_len), dtype="int32")
        #Es por si vamos a agregar informacion de pose "keypoints"
        if args.add_kp:
            #Coordenadas pixel de forma absoluta
            kp_feat = np.zeros((num_ped_in_frame, seq_len, args.kp_num, 3),
                        dtype="float32")
            #desplazamientos
            kp_feat_rel = np.zeros((num_ped_in_frame, seq_len, args.kp_num, 3),
                            dtype="float32")
        #Es por si vamos a agregar informacion del contexto    
        if args.add_social:
            #Maximum number of persons in a frame
            person_max  = args.person_max
            # absolute pixel
            #donde las 3 posiciones son Id_person, x, y
            sourceData = np.zeros((num_ped_in_frame, seq_len, person_max, 3),dtype="float32")

        count_person = 0
        # Aqui se recorre cada una de las personas que hay en toda la secuencia de frames "cur_seq_data"
        for person_id in nf_ped_ids:
            #Se obtiene toda la informacion de persona (Id_person) presente en la secuencia de frames "cur_seq_data
            cur_person_seq = cur_seq_data[cur_seq_data[:, 1] == person_id, :]
            #Este if es para asegurar que todas las personas tengan secuencias de longitud seq_len
            # por que al usar enumerate nos dan cur_seq_data que no son de longitud seq_len
            if len(cur_person_seq) != seq_len:
                # se omite la secuencia que no cubre todos los frames
                continue
            # -----------------------------------------------------------------------
            # AQUI SE VERIFICA QUE LAS PRIMERAS 8 POSICIONES NO SEAN IGUALES
            # -----------------------------------------------------------------------      
            #if(num_ped_in_frame>max_lon):
            #    max_lon=num_ped_in_frame
            # AQUI AGREGAMOS LA INFORMACION DE CONTEXTO SOCIAL
            # Creamos lista con las personas de cada frame
            # esto es con el fin de tener la informacion de todos sus vecinos de la persona_id actual 
            # esto se inicializa cada que cambiamos de una sucesion de seq_len

            if args.add_social:

                con_iguales=0
                for n in range(obs_len-1):
                    if((cur_person_seq[n,2]==cur_person_seq[n+1,2]) and (cur_person_seq[n,3]==cur_person_seq[n+1,3])):
                        con_iguales +=1
                if (con_iguales==obs_len-1):
                    continue

                # Lista de los Id_person de las personas que estan en esa secuencia de frames
                pedID_list = list(nf_ped_ids)
                #Es para guardar la informacion de los vecinos de la persona(person_id) 
                vecinos_person_seq = np.zeros((seq_len, person_max, 3),dtype="float32")
                seq = 0
                # El For va sobre todos los frames que conforman la secuencia de frames
                for num_frame in np.unique(cur_seq_data[:,0]).tolist():
                    # Toda la informacion del frame "num_frame"
                    sseq_frame_data = cur_seq_data[cur_seq_data[:,0] == num_frame, :]
                    # Se toma los Id, x, y del frame "num_frame"	
                    sseq_frame_data = sseq_frame_data[:,1:4]
                    # El For va sobre pedID_list
                    for ped in range( num_ped_in_frame):
                        # Obtenemos el Id de la persona  
                        pedID = pedID_list[ped]
                        #En contramos la informacion de la persona 
                        sped = sseq_frame_data[sseq_frame_data[:, 0] == pedID, :]
                        # Verificamos si hubo informacion de ese pedID, si hay lo agregamos
                        if sped.size != 0:
                            #sourceData[count_person,seq, ped, :] = sped
                            vecinos_person_seq[seq,ped,:] = sped
                    seq+=1
                sourceData[count_person,:,:,:] = vecinos_person_seq
            #[seq_len,2], tiene las coordenadas x,y de formaa absoluta
            cur_person_seq = cur_person_seq[:, 2:]
            cur_person_seq_rel = np.zeros_like(cur_person_seq)
            # first frame is zeros x,y
            cur_person_seq_rel[1:, :] = cur_person_seq[1:, :] - cur_person_seq[:-1, :]

            # es la informacion de x,y y los desplazamientos de todas las person_id que hubieron 
            # en cada secuencia de frames

            cur_seq[count_person, :, :] = cur_person_seq
            cur_seq_rel[count_person, :, :] = cur_person_seq_rel


            #La lista de frames que tiene cada person_id que pertenece a esta secuencia de frames
            frame_idxs = frameList[idx:idx+seq_len]
            
            # Por cada person_id en la secuencia de frames guardamos la secuencia de frames
            # obviamente todas las personas que esten en una misma secuencia de frames, van a tener los
            # la misma lista de frames
            cur_seq_frame[count_person, :] = frame_idxs

            # tiene el person_id de cada una de las personas que si tuvieron secuencias de longitud seq_len
            key_idx.append(person_id)

            # si agregamos informacion de keypoints
            if args.add_kp:
                # get the kp feature from starting frame to seq_len frame)
                # key_idx.append(person_id)
                for i, frame_idx in enumerate(frame_idxs):
                    #print(frame_idx)
                    key = "%d_%d" % (frame_idx, person_id)
                    #print(key)
                    # ignore the kp logits
                    kp_feat[count_person, i, :, :] = kp_feats[key][:, :3] 
                
                # puse un 1 por que al inicio no se a movido asi que si estamos seguro de 
                # que el desplazamiento del peaton sea cero en todos sus keypoints
                #kp_feat_rel[count_person, 0, :, 3]= 1.0

                kp_feat_rel[count_person, 1:, :, :2] = kp_feat[count_person, 1:, :, :2] - kp_feat[count_person, :-1, :, :2]
                kp_feat_rel[count_person, 1:, :,  2] = kp_feat[count_person, 1:, :,  2] * kp_feat[count_person, :-1, :, 2]
                kp_feat_rel[count_person,  0, :,  2] = np.ones((18,))
            count_person += 1
        # El numero de personas que cumplieron que su secuencia fuera de longitud seq_len
        # de la secuencia de frames
        num_person_in_start_frame.append(count_person)

        # only count_person data is preserved
        seq_list.append(cur_seq[:count_person])
        seq_list_rel.append(cur_seq_rel[:count_person])
        seq_frames.append(cur_seq_frame[:count_person])

        # other features
        if args.add_kp:
            kp_list.append(kp_feat[:count_person])
            kp_list_rel.append(kp_feat_rel[:count_person])
        #print(count_person)
        if args.add_social:
            seq_list_person.append(sourceData[:count_person])
            #seq_list_person_rel.append(sourceData_rel[:count_person])
        #hasta aqui 

    #print(" maximo numero de personas que permanecen en toda una secuencia de frames")
    #print(max_p)
    #print("max_person long")
    #print(max_lon)

    # N is numero de secuencias de frames  for each video, K is num_person in each frame
    # el numero total que tendremos es el numero total de personas que hayan cumplido que si tienen secuencia
    seq_list = np.concatenate(seq_list, axis=0)
    seq_list_rel = np.concatenate(seq_list_rel, axis=0)
    seq_frames = np.concatenate(seq_frames, axis=0)

    print("El numero total de ejemplos es: ",len(seq_list))

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

    seq_start_end = np.array([(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])], dtype="int")

    #print(obs_person[0][0])
    # save the data
    data = {
        "obs_traj": obs_traj,
        "obs_traj_rel": obs_traj_rel,
        "pred_traj": pred_traj,
        "pred_traj_rel": pred_traj_rel,
        "key_idx": np.array(key_idx),
        "obs_frames": frames_obs
        #"good_idx": np.array(seq_list_person)
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
            "obstacles":obstacles_world,
        })
    return data





def process_file_modif_varios_inter(data_dirs, list_max_person, args, delim, lim=[]):

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

    # este conjunto va guardando los id de las personas que si tuvieron secuencias de la longitud deseada
    #key_idx    = []

    # va guardando los frames de cada secuencia buena
    seq_frames = []  # [N, seq_len]

    num_person_in_start_frame=[]

    seq_list  = []
    seq_list_rel = []

    kp_list   = []  # [N, seq_len, 18, 2]
    kp_list_rel = []
    
    #seq_list_person=[] 
    todo_flujo = []
    
    for indi,directory in enumerate(used_data_dirs):
        
        seq_list_person_indi = []
        key_idx_indi         = []
        seq_list_indi        = []
        #se obtiene el nombre del archivo sin importar del punto txt, csv,etc.
        #name_sub_data = os.path.splitext(os.path.basename(sub_data))[0]
        #print(name_sub_data)
        sub_data = os.path.join(directory, 'mundo/mun_pos.csv')
        print(sub_data)
        
        if args.obstacles:
            t = directory.split('/')
            data_paths = t[0]+'/'+t[1]+'/'
            dataset_name = t[2]
            obstacles_world = load_world_obstacle_polygons(data_paths,dataset_name)

        kp_feats = {} # "frameidx_personId"
        if args.add_kp:
            kp_file_path = os.path.join(directory,'kp_box.csv')
            with open(kp_file_path, "r") as f:

                for line in f:
                    fidxykp = line.strip().split(delim)
                    key = fidxykp[0] + "_" +fidxykp[1]
                    kp_feats[key] = np.array(fidxykp[2:]).reshape(args.kp_num,3)

        # Trayectory coordinates
        data = np.genfromtxt(sub_data, delimiter= delim)
        # uniques frames
        frameList = np.unique(data[:, 0]).tolist()
        frame_data = [] # people in frame

        for frame in frameList:
            frame_data.append(data[data[:, 0]==frame, :])

        #contador=0
        for idx, frame in enumerate(frameList):
            #la secuencia de frames de size seq_len=obs+pred
            cur_seq_data = frame_data[idx:idx + seq_len]

            nf_ped_ids = reduce(set.intersection,
                                   [set(nf_ped_ids[:,1]) for nf_ped_ids in
                                   cur_seq_data])

            nf_ped_ids = sorted(list(nf_ped_ids))

            cur_seq_data = np.concatenate(cur_seq_data,axis=0)

    
            # El numero de personas de la interseccion de los id_person de cur_seq_data
            num_ped_in_frame = len(nf_ped_ids)

            # los siguientes dos array tienen la misma forma
            # tiene toda la informacion de todas las personas que hay en la secuencia de frames
            # y la informacion que habra sera de x,y de manera absoluta
            # de forma absoluta (sin ninguna transformacion)

            cur_seq = np.zeros((num_ped_in_frame, seq_len, 2), dtype="float32")
            # Este array llevara toda la informacion de todas las personas que hay en esta secuencia
            # de frames, y lo que habra sera sus desplazamientos
            # por ejemplo la primera posicion del la secuencia de frames todostienen desplazamiento cero
            # por que no han avanzado

            cur_seq_rel = np.zeros((num_ped_in_frame, seq_len, 2), dtype="float32")
            # Es el array que tienen la secuencia de Id de todas las personas que hay
            # en las secuencia de frames
            cur_seq_frame = np.zeros((num_ped_in_frame, seq_len), dtype="int32")

            
            if args.add_kp:
                # absolute pixel
                kp_feat = np.zeros((num_ped_in_frame, seq_len, args.kp_num, 3),
                        dtype="float32")
                # velocity
                kp_feat_rel = np.zeros((num_ped_in_frame, seq_len, args.kp_num, 3),
                            dtype="float32")

            if args.add_social:
                person_max = list_max_person[indi]
                # absolute pixel
                sourceData = np.zeros((num_ped_in_frame, seq_len, person_max, 3),dtype="float32")

            # se inicializa cada que cambiamos de secuencia de frames
            count_person = 0
            # Aqui se recorre cada una de las personas que hay en toda la secuencia de frames
            for person_id in nf_ped_ids:
                # se obtiene toda la informacion de persona person_id presente en la secuencia de frames
                cur_person_seq = cur_seq_data[cur_seq_data[:, 1] == person_id, :]

                # este if es para asegurar que todas las personas tengan secuencias de longitud seq_len

                if len(cur_person_seq) != seq_len:
                    # se omite la secuencia que no cubre todos los frames
                    continue

                # contador+=1 
                # -----------------------------------------------------------------------
                # Creamos lista con las personas de cada frame
                # esto es con el fin de tener la informacion de todos sus vecinos de la persona_id actual
                # esto se inicializa cada que cambiamos de una sucesion de seq_len
                if args.add_social:

                    con_iguales = 0
                    for n in range(obs_len-1):
                        if((cur_person_seq[n,2]==cur_person_seq[n+1,2]) and (cur_person_seq[n,3]==cur_person_seq[n+1,3])):
                            con_iguales +=1
                    if(con_iguales==obs_len-1):
                        continue

                    #cur_frame_seq = []
                    #Recorremos cada uno de los frames en los que esta la person_id actual
                    pedID_list = list(nf_ped_ids)

                    #print("secuencia")
                    #print(np.unique(cur_seq_data[:,0]).tolist())
                    # recorremos cada uno de los frames de la secuencia
                    vecinos_person_seq = np.zeros((seq_len, person_max, 3),dtype="float32")

                    seq = 0
                    for num_frame in np.unique(cur_seq_data[:,0]).tolist():
                        # Toda la informacion del frame num_frame
                        sseq_frame_data = cur_seq_data[cur_seq_data[:,0] == num_frame, :]
                        # solo se toman los Id,x,y de toda  informacion
                        sseq_frame_data = sseq_frame_data[:,1:4]
                        for ped in range(num_ped_in_frame):
                            pedID = pedID_list[ped]
                            sped = sseq_frame_data[sseq_frame_data[:, 0] == pedID, :]
                            if sped.size != 0:
                                #sourceData[count_person,seq, ped, :] = sped
                                vecinos_person_seq[seq,ped,:] = sped
                        seq+=1
                    sourceData[count_person,:,:,:] = vecinos_person_seq
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
                frame_idxs = frameList[idx:idx+seq_len]

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
                    kp_feat_rel[count_person, 0, :,   2] = np.ones((18,))
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
            
            
            #Otras caracteristicas
            if args.add_kp:
                kp_list.append(kp_feat[:count_person])
                kp_list_rel.append(kp_feat_rel[:count_person])
            #print(count_person)
            if args.add_social:
                #seq_list_person.append(sourceData[:count_person])
                seq_list_person_indi.append(sourceData[:count_person])
        #aquiiiiiiiiiiiiiiiiiii
        if args.add_social:
            # la informacion de los vecinos
            seq_list_person_indi = np.concatenate(seq_list_person_indi, axis = 0)
            
            obs_person = seq_list_person_indi[:,:obs_len,:,:]
            pred_person = seq_list_person_indi[:,obs_len:,:,:]
            
            seq_list_indi =np.concatenate(seq_list_indi,axis=0)
            print("The numbers of examples is: ", len(seq_list_indi))
            
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
            #if(indi==0):
            #    print(flujo)

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
            "obs_flujo": todo_flujo,
        })

    return data
    

    

    





