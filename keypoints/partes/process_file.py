import os
from tqdm import tqdm
import glob
import numpy as np

# Funcion para preprocesar los datos cargados.
# Esta funcion solo procesa los datos de un conjunto de datos que estan en un solo archivo
def process_file_modif(path_file, args, delim):

  obs_len = args.obs_len
  pred_len = args.pred_len
  seq_len = obs_len + pred_len

  data = []
  num_person_in_start_frame = []

  seq_list = []
  seq_list_rel = []
  seq_frames = []

  kp_list = []  # [N, seq_len, 17, 2]
  kp_list_rel = []
  
  
  #Estas listas tienen la informacion de frames de todas las sucesiones de tamano seq_len
  seq_list_person = []
  seq_list_person_rel = []

  #Load other features if necessary
  #Esta lista tendra los Id_person de las personas de cada secucuencia seq_len que se puedan hacer
  key_idx = []
  kp_feats = {}  # "frameidx_personId"
  
  # Si queremos agregar keypoints que se encuantran en un txt
  if args.add_kp:
    
    with open(args.kp_path, "r") as f:
      for line in f:
        fidxykp = line.strip().split(delim)
        key = fidxykp[0] + "_" +fidxykp[1]
        #key_idx.append(key)
        kp_feats[key] = np.array(fidxykp[2:]).reshape(args.kp_num,3) # TODO: MOdificar segun la prob
  
  #Leemos la informacion de las coordenadas
  with open(path_file, "r") as traj_file:
    for line in traj_file:
      #fidx, pid, x, y = line.strip().split('\t')
      fidx, pid, x, y = line.strip().split(delim)
      data.append([fidx, pid, x, y])
  data = np.array(data, dtype="float32")

  # asumimos que los id de los frames estan ordenados de forma ascendente
  frames = np.unique(data[:, 0]).tolist()  # todos los id_frames de forma unica
  frame_data = []  #people in frame 

  
  #Toda la informacion de cada frame
  #id_frame, id_person, x, y 
  for frame in frames:
    frame_data.append(data[data[:, 0]==frame, :])


  # Es para decir la cantidad de personas maximas en una secuencia de frames
  person_max = args.person_max #---  22
    
    #num_c_frame = []
    # el numero de personas por frame
    #num_c_frame.append(data[data[:, 0]==frame, :].shape[0])
  #person_max = np.max(num_c_frame)

  #print('Numero de maximo de personas que hay en un frame')
  #print(person_max)
  #print(len(num_c_frame))
  #max_p=0
  #max_lon=0

  for idx, frame in enumerate(frames):
    
    #La secuencia de frames de size seq_len = obs+pred
    #id_frame, id_person, x, y para cada persona presente en el frame
    cur_seq_data = np.concatenate(frame_data[idx:idx + seq_len], axis = 0)

    #Los indices unicos de las personas en toda la secuencia de frames "cur_seq_data"
    persons_in_cur_seq = np.unique(cur_seq_data[:,1])
    
    #El numero de personas unicas que hay en la secuencia de frames "cur_seq_data"
    num_person_in_cur_seq = len(persons_in_cur_seq)

    #print("numero de personas")
    #print(num_person_in_cur_seq)

    #if(num_person_in_cur_seq>max_p):
    #  max_p=num_person_in_cur_seq
 
    #Los siguientes dos array tienen la misma forma
    #"cur_seq" Contiene toda la informacion de todas las personas que hay en la secuencia de frames 
    #y la informacion que habra sera de x,y de forma absoluta(sin ninguna transformacion)
    cur_seq = np.zeros((num_person_in_cur_seq, seq_len, 2), dtype="float32")

    #Este array llevara toda la informacion de todas las personas que hay en esta secuencia 
    #de frames, y lo que habra sera sus desplazamientos, por ejemplo la primera posicion de la secuencia de frames todos tienen desplazamiento cero
    #por que no han avanzado
    cur_seq_rel = np.zeros((num_person_in_cur_seq, seq_len, 2), dtype="float32")
  
    #Es el array que tienen la secuencia de Id_person de todas las personas que hay en las secuencia de frames
    cur_seq_frame = np.zeros((num_person_in_cur_seq, seq_len), dtype="int32")
    
    
    
    #Es por si vamos a agregar informacion de pose "keypoints"
    if args.add_kp:
      #Coordenadas pixel de forma absoluta
      kp_feat = np.zeros((num_person_in_cur_seq, seq_len, args.kp_num, 2),
                        dtype="float32")
      #desplazamientos
      kp_feat_rel = np.zeros((num_person_in_cur_seq, seq_len, args.kp_num, 2),
                            dtype="float32")

    #Es por si vamos a agregar informacion del contexto    
    if args.add_social:
      
      # absolute pixel
      #donde las 3 posiciones son Id_person, x, y
      sourceData = np.zeros((num_person_in_cur_seq, seq_len, person_max, 3),dtype="float32")
      
      # velocity
      #sourceData_rel = np.zeros((num_person_in_cur_seq, seq_len, person_max, 3),dtype="float32")
    
    # Se inicializa cada que cambiamos de secuencia de frames
    count_person = 0

    # Aqui se recorre cada una de las personas que hay en toda la secuencia de frames "cur_seq_data"
    for person_id in persons_in_cur_seq:
        
      #Se obtiene toda la informacion de persona (Id_person) presente en la secuencia de frames "cur_seq_data"
      cur_person_seq = cur_seq_data[cur_seq_data[:, 1] == person_id, :]
      
      #Este if es para asegurar que todas las personas tengan secuencias de longitud seq_len
      if len(cur_person_seq) != seq_len:
        # se omite la secuencia que no cubre todos los frames
        continue
      


      # -----------------------------------------------------------------------
      # AQUI SE VERIFICA QUE LAS PRIMERAS 8 POSICIONES NO SEAN IGUALES
      # -----------------------------------------------------------------------
      con_iguales=0
      for n in range(obs_len-1):
        if((cur_person_seq[n,2]==cur_person_seq[n+1,2]) and (cur_person_seq[n,3]==cur_person_seq[n+1,3])):
          con_iguales +=1
      if(con_iguales==obs_len-1):
        continue

      #if(num_person_in_cur_seq>max_lon):
      #  max_lon=num_person_in_cur_seq
      # AQUI AGREGAMOS LA INFORMACION DE CONTEXTO SOCIAL
      # Creamos lista con las personas de cada frame
      # esto es con el fin de tener la informacion de todos sus vecinos de la persona_id actual 
      # esto se inicializa cada que cambiamos de una sucesion de seq_len
      if args.add_social:
        #cur_frame_seq = []

        # Lista de los Id_person de las personas que estan en esa secuencia de frames
        pedID_list = list(persons_in_cur_seq)
        
        #Es para guardar la informacion de los vecinos de la persona(person_id) 
        vecinos_person_seq = np.zeros((seq_len, person_max, 3),dtype="float32")

        seq = 0
        # El For va sobre todos los frames que conforman la secuencia de frames
        for num_frame in np.unique(cur_seq_data[:,0]).tolist():

          # Toda la informacion del frame "num_frame"
          sseq_frame_data = cur_seq_data[cur_seq_data[:,0] == num_frame, :]
          
          # Se toma los Id, x, y del frame "num_frame"
          sseq_frame_data = sseq_frame_data[:,1:4]
          
          # El For va sobre el numero de personas unicas que hay en la secuencia de frames
          for ped in range(num_person_in_cur_seq):
            
            # Obtenemos el Id de la persona  
            pedID = pedID_list[ped]
            # Esto hay que quitarlo
            if pedID == 0:
              continue
            else:
              #En contramos la informacion de la persona 
              sped = sseq_frame_data[sseq_frame_data[:, 0] == pedID, :]

              # Verificamos si hubo informacion de ese pedID, si hay lo agregamos
              if sped.size != 0:
                #sourceData[count_person,seq, ped, :] = sped
                vecinos_person_seq[seq,ped,:] = sped
                
          seq+=1
        
        
        #vecinos_person_seq_rel = np.zeros_like(vecinos_person_seq)
        
        
        #vecinos_person_seq_rel[:,:,0] = vecinos_person_seq[:,:,0]
        #vecinos_person_seq_rel[1:,:,1:] = vecinos_person_seq[1:,:,1:]-vecinos_person_seq[:-1,:,1:]
        
        sourceData[count_person,:,:,:] = vecinos_person_seq
        #sourceData_rel[count_person,:,:,:] = vecinos_person_seq_rel
        
            
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
      frame_idxs = frames[idx:idx+seq_len]
    
      # Por cada person_id en la secuencia de frames guardamos la secuencia de frames
      #obviamente todas las personas que esten en una misma secuencia de frames, van a tener los
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
          kp_feat[count_person, i, :, :] = kp_feats[key][:, :2]
        
        kp_feat_rel[count_person, 1:, :, :] = \
            kp_feat[count_person, 1:, :, :] - kp_feat[count_person, :-1, :, :]
    
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
     
  #print("personas max en la secuencia de frames")
  #print(max_p)
  #print("max_person long")
  #print(max_lon)
  
# N is numero de secuencias de frames  for each video, K is num_person in each frame
  # el numero total que tendremos es el numero total de personas que hayan cumplido que si tienen secuencia
  seq_list = np.concatenate(seq_list, axis=0)
  seq_list_rel = np.concatenate(seq_list_rel, axis=0)
  seq_frames = np.concatenate(seq_frames, axis=0)

  print("El numero total de ejemplos")
  print(len(seq_list))
  
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
    
  return data





#Esta funcion procesa los sub-archivos
#Es decir de un archivo selecciono ciertos conjuntos de frame y por lo tanto 
#Se hacen archivos por cada conjunto de frames

def process_file_modif_varios(path_file, args, delim):
    
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
    
    seq_list_person=[]
    seq_list_person_rel=[]
    
    # aqui tengo todas las direcciones de los sub_datas
    sub_datas = glob.glob(os.path.join(path_file,"*.csv"))
    print(sub_datas)
    
    for sub_data in tqdm(sub_datas, ascii=True):
        
        #se obtiene el nombre del archivo sin importar del punto txt, csv,etc.
        name_sub_data = os.path.splitext(os.path.basename(sub_data))[0]
        print(name_sub_data)


        kp_feats = {} # "frameidx_personId"
        if args.add_kp:
            kp_file_path= os.path.join(args.kp_path,"%s.csv" % name_sub_data)
            
            with open(kp_file_path, "r") as f:
    
                for line in f:
                    fidxykp = line.strip().split(delim)
                    key = fidxykp[0] + "_" +fidxykp[1]
                    kp_feats[key] = np.array(fidxykp[2:]).reshape(args.kp_num,3) # TODO: MOdificar segun la prob 
  
        data=[]
        with open(sub_data, "r") as traj_file:
            for line in traj_file:
                #fidx, pid, x, y = line.strip().split('\t')
                fidx, pid, x, y = line.strip().split(delim)
                data.append([fidx, pid, x, y])
        data = np.array(data, dtype="float32")

        frames = np.unique(data[:, 0]).tolist()  # all frame_idx
        frame_data = []  # people in frame
        #num_c_frame=[]
        #toda la informacion de cada frame
        for frame in frames:
            frame_data.append(data[frame == data[:, 0], :])
            # el numero de personas por frame
            #num_c_frame.append(data[frame == data[:, 0], :].shape[0])
        #person_max=np.max(num_c_frame)
        person_max=8
        #print('Numero de maximo de personas que hay en un frame')
        #print(np.max(num_c_frame))
        #print(person_max)
        contador=0
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
            
            # se inicializa cada que cambiamos de secuencia de frames
            count_person = 0
            
            if args.add_kp:
                # absolute pixel
                kp_feat = np.zeros((num_person_in_cur_seq, seq_len, args.kp_num, 2),
                        dtype="float32")
                # velocity
                kp_feat_rel = np.zeros((num_person_in_cur_seq, seq_len, args.kp_num, 2),
                            dtype="float32")
                
            if args.add_social:
                # absolute pixel
                sourceData = np.zeros((num_person_in_cur_seq, seq_len, person_max, 3),dtype="float32")
                
                # velocity
                sourceData_rel = np.zeros((num_person_in_cur_seq, seq_len, person_max, 3),dtype="float32")
                
            # Aqui se recorre cada una de las personas que hay en toda la secuencia de frames
            for person_id in persons_in_cur_seq:
                # se obtiene toda la informacion de persona person_id presente en la secuencia de frames
                cur_person_seq = cur_seq_data[cur_seq_data[:, 1] == person_id, :]
                
                # este if es para asegurar que todas las personas tengan secuencias de longitud seq_len
                

                if len(cur_person_seq) != seq_len:
                    # se omite la secuencia que no cubre todos loa frames
                    continue

                con_iguales=0
                for n in range(obs_len-1):
                    if((cur_person_seq[n,2]==cur_person_seq[n+1,2]) and (cur_person_seq[n,3]==cur_person_seq[n+1,3])):
                        con_iguales +=1
                if(con_iguales==obs_len-1):
                    continue

                contador +=1

                # -----------------------------------------------------------------------
                # Creamos lista con las personas de cada frame
                # esto es con el fin de tener la informacion de todos sus vecinos de la persona_id actual 
                # esto se inicializa cada que cambiamos de una sucesion de seq_len
                if args.add_social:
                    #cur_frame_seq = []
                    #Recorremos cada uno de los frames en los que esta la person_id actual
                    pedID_list = list(persons_in_cur_seq)
                    
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
                        for ped in range(num_person_in_cur_seq):
                            pedID = pedID_list[ped]
                            if pedID == 0:
                                print('continueeeeeeeeeee---------')
                                continue
                            else:
                                sped = sseq_frame_data[sseq_frame_data[:, 0] == pedID, :]
                                if sped.size != 0:
                                    #sourceData[count_person,seq, ped, :] = sped
                                    vecinos_person_seq[seq,ped,:] = sped
                        seq+=1
                        
                    vecinos_person_seq_rel = np.zeros_like(vecinos_person_seq)
                    #aqui agregamos solo los IDs
                    vecinos_person_seq_rel[:,:,0] = vecinos_person_seq[:,:,0]
                    vecinos_person_seq_rel[1:,:,1:] = vecinos_person_seq[1:,:,1:]-vecinos_person_seq[:-1,:,1:]
                    
                    sourceData[count_person,:,:,:] = vecinos_person_seq
                    sourceData_rel[count_person,:,:,:] = vecinos_person_seq_rel
                    
                    #seq_list_person_obs.append( sourceData[:obs_len,:,:] )
                    #seq_list_person_pred.append( sourceData[obs_len:,:,:] )
                
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
                        kp_feat[count_person, i, :, :] = kp_feats[key][:, :2]
                    kp_feat_rel[count_person, 1:, :, :] = kp_feat[count_person, 1:, :, :] - kp_feat[count_person, :-1, :, :]
                count_person += 1

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
            #print(count_person)
            if args.add_social:
                seq_list_person.append(sourceData[:count_person])
                #seq_list_person_rel.append(sourceData_rel[:count_person])
      
        print(contador)       
    num_seq = len(seq_list)  # el numero total se secuencias de frames de tamano seq_len que hubieron,sea 
    
    
    
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
        "key_idx": np.array(key_idx),
        "frames_obs": frame_obs,
        "frames_pred": frame_pred,
        #"frame_list":seq_frameidx_list
    }
    
    #obs_person = np.array(seq_list_person_obs)
    #pred_person = np.array(seq_list_person_pred)
    
    if args.add_social:
        seq_list_person = np.concatenate(seq_list_person, axis = 0)
        seq_list_person_rel = np.concatenate(seq_list_person_rel, axis = 0)
        
        #pred_person = np.array(seq_list_person_pred)
        obs_person = seq_list_person[:,:obs_len,:,:]
        pred_person = seq_list_person[:,obs_len:,:,:]
        obs_person_rel = seq_list_person_rel[:,:obs_len,:,:]
        pred_person_rel = seq_list_person_rel[:,obs_len:,:,:]
        
        data.update({
            "obs_person": obs_person,
            "pred_person": pred_person,
            "obs_person_rel": obs_person_rel,
            "pred_person_rel": pred_person_rel,
        })
    
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
    
    return data

def datos_subprocesados_varios(datasets, args, delim):

    data_dirs = ['../data1/eth/univ/mundo', '../data1/eth/hotel/mundo',
                 '../data1/ucy/zara/zara01/mundo', '../data1/ucy/zara/zara02/mundo',
                 '../data1/ucy/univ/mundo']
        
    name_flujo = ['eth_univ','eth_hotel','zara01','zara02','ucyuniv']

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
    
    seq_list_person=[]
    seq_list_person_rel=[]
    
    # aqui tengo todas las direcciones de los sub_datas
    #sub_datas = glob.glob(os.path.join(path_file,"*.csv"))
    #print(sub_datas)
    
    for indi,directory in enumerate(used_data_dirs):
        name_sub_flujo ='flujo_bien_non_'+ used_name[indi]+'_total_mundo_12.npy'

        directory_sub_flujo = os.path.join(args.directory_flujo,name_sub_flujo)

        sub_flujo = np.load(directory_sub_flujo)
        if indi==0:
            todo_flujo = sub_flujo 
        if indi>0:
            todo_flujo = np.concatenate((todo_flujo,sub_flujo),axis=0)
           

        #se obtiene el nombre del archivo sin importar del punto txt, csv,etc.
        #name_sub_data = os.path.splitext(os.path.basename(sub_data))[0]
        #print(name_sub_data)
        sub_data= os.path.join(directory, 'mun_pos.csv')
        print(sub_data)

        kp_feats = {} # "frameidx_personId"
        if args.add_kp:
            kp_file_path= os.path.join(directory,'keypoints.csv')
            
            with open(kp_file_path, "r") as f:
    
                for line in f:
                    fidxykp = line.strip().split(delim)
                    key = fidxykp[0] + "_" +fidxykp[1]
                    kp_feats[key] = np.array(fidxykp[2:]).reshape(args.kp_num,3) # TODO: MOdificar segun la prob 
  
        data=[]
        with open(sub_data, "r") as traj_file:
            for line in traj_file:
                #fidx, pid, x, y = line.strip().split('\t')
                fidx, pid, x, y = line.strip().split(delim)
                data.append([fidx, pid, x, y])
        data = np.array(data, dtype="float32")

        frames = np.unique(data[:, 0]).tolist()  # all frame_idx
        frame_data = []  # people in frame
        #num_c_frame=[]
        #toda la informacion de cada frame
        for frame in frames:
            frame_data.append(data[frame == data[:, 0], :])
            # el numero de personas por frame
            #num_c_frame.append(data[frame == data[:, 0], :].shape[0])
        #person_max=np.max(num_c_frame)
        person_max=8
        #print('Numero de maximo de personas que hay en un frame')
        #print(np.max(num_c_frame))
        #print(person_max)
        contador=0
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
            
            # se inicializa cada que cambiamos de secuencia de frames
            count_person = 0
            
            if args.add_kp:
                # absolute pixel
                kp_feat = np.zeros((num_person_in_cur_seq, seq_len, args.kp_num, 2),
                        dtype="float32")
                # velocity
                kp_feat_rel = np.zeros((num_person_in_cur_seq, seq_len, args.kp_num, 2),
                            dtype="float32")
                
            if args.add_social:
                # absolute pixel
                sourceData = np.zeros((num_person_in_cur_seq, seq_len, person_max, 3),dtype="float32")
                
                # velocity
                sourceData_rel = np.zeros((num_person_in_cur_seq, seq_len, person_max, 3),dtype="float32")
                
            # Aqui se recorre cada una de las personas que hay en toda la secuencia de frames
            for person_id in persons_in_cur_seq:
                # se obtiene toda la informacion de persona person_id presente en la secuencia de frames
                cur_person_seq = cur_seq_data[cur_seq_data[:, 1] == person_id, :]
                
                # este if es para asegurar que todas las personas tengan secuencias de longitud seq_len
                

                if len(cur_person_seq) != seq_len:
                    # se omite la secuencia que no cubre todos loa frames
                    continue

                con_iguales=0
                for n in range(obs_len-1):
                    if((cur_person_seq[n,2]==cur_person_seq[n+1,2]) and (cur_person_seq[n,3]==cur_person_seq[n+1,3])):
                        con_iguales +=1
                if(con_iguales==obs_len-1):
                    continue

                contador +=1

                # -----------------------------------------------------------------------
                # Creamos lista con las personas de cada frame
                # esto es con el fin de tener la informacion de todos sus vecinos de la persona_id actual 
                # esto se inicializa cada que cambiamos de una sucesion de seq_len
                if args.add_social:
                    #cur_frame_seq = []
                    #Recorremos cada uno de los frames en los que esta la person_id actual
                    pedID_list = list(persons_in_cur_seq)
                    
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
                        for ped in range(num_person_in_cur_seq):
                            pedID = pedID_list[ped]
                            if pedID == 0:
                                print('continueeeeeeeeeee---------')
                                continue
                            else:
                                sped = sseq_frame_data[sseq_frame_data[:, 0] == pedID, :]
                                if sped.size != 0:
                                    #sourceData[count_person,seq, ped, :] = sped
                                    vecinos_person_seq[seq,ped,:] = sped
                        seq+=1
                        
                    vecinos_person_seq_rel = np.zeros_like(vecinos_person_seq)
                    #aqui agregamos solo los IDs
                    vecinos_person_seq_rel[:,:,0] = vecinos_person_seq[:,:,0]
                    vecinos_person_seq_rel[1:,:,1:] = vecinos_person_seq[1:,:,1:]-vecinos_person_seq[:-1,:,1:]
                    
                    sourceData[count_person,:,:,:] = vecinos_person_seq
                    sourceData_rel[count_person,:,:,:] = vecinos_person_seq_rel
                    
                    #seq_list_person_obs.append( sourceData[:obs_len,:,:] )
                    #seq_list_person_pred.append( sourceData[obs_len:,:,:] )
                
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
                        kp_feat[count_person, i, :, :] = kp_feats[key][:, :2]
                    kp_feat_rel[count_person, 1:, :, :] = kp_feat[count_person, 1:, :, :] - kp_feat[count_person, :-1, :, :]
                count_person += 1

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
            #print(count_person)
            if args.add_social:
                seq_list_person.append(sourceData[:count_person])
                #seq_list_person_rel.append(sourceData_rel[:count_person])
      
        print(contador)       
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
        "key_idx": np.array(key_idx),
        "frames_obs": frame_obs,
        "frames_pred": frame_pred,
        #"frame_list":seq_frameidx_list
    }
    
    #obs_person = np.array(seq_list_person_obs)
    #pred_person = np.array(seq_list_person_pred)
    
    if args.add_social:
        seq_list_person = np.concatenate(seq_list_person, axis = 0)
        seq_list_person_rel = np.concatenate(seq_list_person_rel, axis = 0)
        
        #pred_person = np.array(seq_list_person_pred)
        obs_person = seq_list_person[:,:obs_len,:,:]
        pred_person = seq_list_person[:,obs_len:,:,:]
        obs_person_rel = seq_list_person_rel[:,:obs_len,:,:]
        pred_person_rel = seq_list_person_rel[:,obs_len:,:,:]
        
        data.update({
            "obs_person": obs_person,
            "pred_person": pred_person,
            "obs_person_rel": obs_person_rel,
            "pred_person_rel": pred_person_rel,
        })
    
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


