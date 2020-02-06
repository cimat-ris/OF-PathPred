import matplotlib.pyplot as plt
import numpy as np
import math

def vector_ortonormal(vec_dir):
    """ 
      Funcion para calcular el vector normal
      Recibe: un vector de la forma (x,y)
      Retorna: el vector normal del recibido 
    """
    return np.array([vec_dir[1],-vec_dir[0]])

def vectores_direccion(data):
    """
     Esta funcion recibe una matriz [obs_len,2]
     el cual es la trayectoria del observador

     Recibe: La matriz data [obs_len, 2]
     
     Retorna: Un matriz direcciones de la forma [obs_len, 2] 
    """
    pasos = len(data)
    direcciones = np.zeros((pasos,2),dtype='float')

    for i in range(pasos):

        if(i==0 or i==1):
            if((data[0][0]!=data[1][0]) or (data[0][1]!=data[1][1]) ):
                direcciones[i] = np.array([data[1][0]-(data[0][0]), data[1][1]-(data[0][1])])
            else:
                for j in range(i,pasos-1):
                    if( (data[j][0]!=data[j+1][0]) or (data[j][1]!=data[j+1][1]) ):
                        direcciones[i] = np.array([data[j+1][0]-(data[j][0]), data[j+1][1]-(data[j][1])])
        else:
            if((data[i][0]!=data[i-1][0]) or (data[i][1]!=data[i-1][1])):
                direcciones[i] = np.array([data[i][0]-(data[i-1][0]), data[i][1]-(data[i-1][1])])
            else:
                # Si es el ultimo 
                if(i==(pasos-1)):
                    for j in range(pasos-2,0,-1):
                        if((data[j][0]!=data[j-1][0]) or (data[j][1]!=data[j-1][1]) ):
                            direcciones[i] = np.array([data[j][0]-(data[j-1][0]), data[j][1]-(data[j-1][1])])
                else:
                    ban=0
                    for j in range(i+1,pasos):
                        if( (data[j][0]!=data[j-1][0]) or (data[j][1]!=data[j-1][1])):
                            direcciones[i] = np.array([data[j][0]-(data[j-1][0]), data[j][1]-(data[j-1][1])])
                            ban=1
                    if(ban==0):
                        for j in range(i-1,0,-1):
                            if((data[j][0]!=data[j-1][0]) or (data[j][1]!=data[j-1][1])):
                                direcciones[i] = np.array([data[j][0]-(data[j-1][0]), data[j][1]-(data[j-1][1])])
                  
    return direcciones

def in_recta(Point_eval, vec_dir, Point_in_recta):
    """ Funcion para verificar si el punto esta en la recta """
    # Positivo es lado derecho
    # Negativo es lado izquierdo
    # Cero si esta en la recta
    return np.sum(Point_eval*vector_ortonormal(vec_dir)) - np.sum(Point_in_recta*vector_ortonormal(vec_dir)) 

def in_cono(Point_eval,vec_dir1,vec_dir2,Point_in_recta):
    """ Funcion para verificar si el punto esta en la seccion del cono """
    # 1 si  pertenece al cono
    # 0 si no pertenece al cono
    
    # Evaluamos si esta en la recta
    resp1 = in_recta(Point_eval,vec_dir1,Point_in_recta)
    resp2 = in_recta(Point_eval,vec_dir2,Point_in_recta)
    if resp1>0 and resp2<0:
        return 1
    return 0

def grafica_vec(ind,i,PPi,vec_dir,list_vect_new=[],Points=[],factor=10000):
    """ Funcion para graficar y visualizar los vectores y puntos"""

    # Graficamos el vector direccion y el normal
    vec_norm = vector_ortonormal(vec_dir)
    plt.plot([PPi[0],PPi[0]+vec_dir[0]*factor],[PPi[1],PPi[1]+vec_dir[1]*factor])
    plt.plot([PPi[0]-vec_norm[0]*factor,PPi[0]+vec_norm[0]*factor],[PPi[1]-vec_norm[1]*factor,PPi[1]+vec_norm[1]*factor])
    plt.axis([-1, 17, -1, 14])
    
    # Graficamos los nuevos vectores de particion
    for vec_new in list_vect_new:
        plt.plot([PPi[0],PPi[0]+vec_new[0]*factor],[PPi[1],PPi[1]+vec_new[1]*factor],'--')
    
    # Graficamos los puntos que queramos
    for vec in Points:
        plt.plot(vec[0], vec[1],'ro') # Test
    name= 'mapa_zara02'+ str(int(ind))+'_'+str(int(i))+'.jpg'
    #plt.savefig("trayectorias.jpg")
    plt.savefig(name)
    plt.show()

def get_partition_cono(vec_dir,theta0,thetaf,num_part):
    """ Funcion para particionar el cono """

    # Calculamos el vector normal de vec_dir
    vec_norm = vector_ortonormal(vec_dir)
    #print("vector normal")
    #print(vec_norm)

    list_vect_new = []
    for theta_i in np.linspace(theta0,thetaf,num_part+1):
        theta2 = theta_i
        theta1 = 90-theta2
        # Calculamos el angulo entre las rectas
        cos1 = np.cos( theta1*np.pi/180.0)
        cos2 = np.cos( theta2*np.pi/180.0)
        # Calculamos la norma del vector
        norma1 = np.linalg.norm(vec_norm)
        norma2 = np.linalg.norm(vec_dir)

        # Calculamos la constante
        c1 = norma1*cos1
        c2 = norma2*cos2

        # Calculamos el nuevo vector
        vec_new =   c1*vec_dir - c2*vec_norm
        list_vect_new.append(vec_new)
        
    return list_vect_new


#------------------------------------------------

def calcular_flujo_optico(PPi, coord , vec_dir_PPi, vec_vel_coord):
    """ Funcion que calcula el flujo optico de un vecino(coord) con respecto a PPi"""
    
    # vector que esta entre PPi y coord (posicion relativa)
    # delta P
    vec_dir_v_PPi = [coord[0]-PPi[0],coord[1]-PPi[1]]

    # Esto es para encontrar la magnitud de proyectar el vector  
    #prod = prod_punto(vec_dir_PPi,vec_dir_v_PPi)
    #n1 = np.linalg.norm(vec_dir_PPi)
    #n2 = np.linalg.norm(vec_dir_v_PPi)
    #tam = (prod)/(n1*n2)
    #X = coord[0]/tam

    # El angulo de rotacion para ponerlo en los ejes en el marco del YO(PPi)
    theta = math.atan2(vec_dir_PPi[1], vec_dir_PPi[0]) 
    #theta = (math.pi/2)+ theta

    # matriz de rotacion
    mr = np.array( [[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])

    # la posicion relativa del vecino con PPi en el marco de PPi
    posicion_rel_marco_ppi = np.array( [mr[0,0]*vec_dir_v_PPi[0]+mr[0,1]*vec_dir_v_PPi[1],mr[1,0]*vec_dir_v_PPi[0]+mr[1,1]*vec_dir_v_PPi[1]])

    # para calcular la velocidad relativa de PPi con Coord en el marco de PPi 
    delta_vel = vec_vel_coord-vec_dir_PPi
   
    delta_vel_rot = np.array([mr[0,0]*delta_vel[0]+mr[0,1]*delta_vel[1],mr[1,0]*delta_vel[0]+mr[1,1]*delta_vel[1]])
    #return((delta_vel_rot[0]-X*delta_vel_rot[1])*tam)

    #x = posicion_rel_marco_ppi[0]/posicion_rel_marco_ppi[1]
    x = posicion_rel_marco_ppi[1]/posicion_rel_marco_ppi[0]
    #flujo= (delta_vel_rot[0]-x*delta_vel_rot[1])*posicion_rel_marco_ppi[1]
    flujo= (delta_vel_rot[1]-x*delta_vel_rot[0])*posicion_rel_marco_ppi[0]
    return flujo

def get_vector_cono_flujo( PPi, vect_dir_PPi , p_veci, vel_p_veci, list_vect_new):
    # nlen es 64
    nlen = len(list_vect_new)-1
    vec_coord = [[]]*nlen

    vec_flujo = np.zeros(nlen)
    tam = len(p_veci)
    vecinos = []

    for i in range(tam):
        coord = p_veci[i]
        for k in range(len(vec_flujo)):
            if in_cono(coord,list_vect_new[k],list_vect_new[k+1],PPi):

                if(vec_flujo[k]!=0):
                            #distancia con el existente
                            d2 =(PPi[0]-vec_coord[k][0])**2+(PPi[1]-vec_coord[k][1])**2
                            d2 = math.sqrt(d2)

                            #distancia con el actual
                            d1 =(PPi[0]-coord[0])**2+(PPi[1]-coord[1])**2
                            d1 =math.sqrt(d1)

                            # si punto actual es mas cercano que el anterior
                            if(d1<d2):
                                vecinos.remove([vec_coord[k][0],vec_coord[k][1]])
                                vel = vel_p_veci[i]
                                u = calcular_flujo_optico(PPi,coord,vect_dir_PPi, vel)
                                vec_flujo[k] = u
                                vec_coord[k] = coord
                         
                                vecinos.append(coord)
                else:
                    vel = vel_p_veci[i]
                    u = calcular_flujo_optico(PPi, coord, vect_dir_PPi, vel)
                    vec_flujo[k] = u
                    vec_coord[k] = coord
                    vecinos.append(coord)
    return vec_flujo,vecinos


def llenado_vector_flujo(p_veci, vel_p_veci,  PPi, vec_dir,ind,i):
    theta0 = 5
    thetaf = 175
    num_part = 64 
    # obtenemos los rayos
    list_vect_new = get_partition_cono( vec_dir, theta0, thetaf, num_part)

    vector_flujo, vecinos = get_vector_cono_flujo(PPi, vec_dir,p_veci,vel_p_veci, list_vect_new)
    # aqui graficamos
    #grafica_vec(ind,i,PPi,vec_dir, list_vect_new, p_veci,  factor=100000)
    return vector_flujo
"""
def calcular_vel_vec(i, sequence_veci, limites):
    #vecino en la posiciion actual
    frame = sequence_veci[i,:,:]

    other_x = (frame[otherpedindex, 1]-(min_x))/((max_x-(min_x))*1.0)
    other_y = (frame[otherpedindex, 2]-(min_y))/((max_x-(min_x))*1.0)

    if(i==0):
        frame_pos = sequence_veci[i+1,:,:]
        other_x_pos = frame_pos[otherpedindex,1]
        other_y_pos = frame_pos[otherpedindex,2]

        if(other_x_pos!=0 or other_y_pos!=0):
            other_x_pos = (other_x_pos-(min_x))/((max_x-(min_x))*1.0)
            other_y_pos = (other_y_pos-(min_y))/((max_x-(min_x))*1.0)
            vel_other = [ other_x_pos-other_x, other_y_pos-other_y]
            return vel_other
        else:
            for j in range(i+2,len(data)-1):
                frame_pos = sequence_veci[j,:,:]
                other_x_pos = frame_pos[otherpedindex,1]
                other_y_pos = frame_pos[otherpedindex,2]

                if(other_x_pos!=0 or other_y_pos!=0):
                    other_x_pos = (other_x_pos-(min_x))/((max_x-(min_x))*1.0)
                    other_y_pos = (other_y_pos-(min_y))/((max_x-(min_x))*1.0)
                    vel_other = [ other_x_pos-other_x, other_y_pos-other_y]
                    return vel_other
    else:
        frame_ant = sequence_veci[i-1,:,:]
        other_x_ant = frame_ant[otherpedindex,1]
        other_y_ant = frame_ant[otherpedindex,2]

        if((other_x_ant!=0) or (other_y_ant!=0)):
            other_x_ant = (other_x_ant-(min_x))/ ((max_x-(min_x))*1.0)
            other_y_ant = (other_y_ant-(min_y))/ ((max_x-(min_x))*1.0)
            vel_other = [other_x-other_x_ant, other_y-other_y_ant]
            return

        else:
            if(i==(len(data)-1)):
                for j in range(len(data)-2,-1,-1):
                    frame_ant =sequence_veci[j,:,:]
                    other_x_ant = frame_ant[otherpedindex,1]
                    other_y_ant = frame_ant[otherpedindex,2]
                    if((other_x_ant!=0) or (other_y_ant!=0) ):
                        other_x_ant = (other_x_ant-(min_x))/ ((max_x-(min_x))*1.0)
                        other_y_ant = (other_y_ant-(min_y))/ ((max_x-(min_x))*1.0)
                        vel_other = [other_x-other_x_ant, other_y-other_y_ant]
            else:
                for j in range(i+1,len(data)):
                    frame_pos = sequence_veci[j,:,:]
                    other_x_pos = frame_pos[otherpedindex,1]
                    other_y_pos = frame_pos[otherpedindex,2]
                    if((other_x_pos!=0) or (other_y_pos!=0)):
                        other_x_pos = (other_x_pos-(min_x))/((max_x-(min_x))*1.0)
                        other_y_pos = (other_y_pos-(min_y))/((max_x-(min_x))*1.0)
                        vel_other = [ other_x_pos-other_x, other_y_pos-other_y]
                    if(j==(len(data)-1)):
                        for j in range(len(data)-2,0,-1):
                                if((data[j][0]!=data[j-1][0]) or (data[j][1]!=data[j-1][1]) ):
                                    direcciones[i]=np.array([data[j][0]-(data[j-1][0]), data[j][1]-(data[j-1][1])])



"""


def vector_secuencia_vecinos(sequence_veci, Id, obs_traj,ind):

    #Calculamos los vectores de direccion
    direcciones = vectores_direccion(obs_traj)
    #print('direcciones')
    #print(direcciones)

    sl = sequence_veci.shape[0]
    mnp = sequence_veci.shape[1]

    sequence_vec_flujo = np.zeros((sl,64), dtype='float')

    for i in range(sl): #Longitud de la secuencia

        # frame es de [mnp,3]  
        frame = sequence_veci[i,:,:]
        
        # saco los vecinos previos o los posteriores
        if(i==0):
            frame_pos = sequence_veci[i+1,:,:]
        else:
            frame_ant = sequence_veci[i-1,:,:]

        # saco toda la informacion de YO(PPi)
        person_sec = frame[frame[:,0]== Id,:][0]

        x_current, y_current = obs_traj[i][0], obs_traj[i][1]

        p_veci = []
        vel_p_veci = []
        
        for otherpedindex in range(mnp):

            # modifique
            if((frame[otherpedindex, 0] == person_sec[0]) or frame[otherpedindex, 0]==0 ):
                # Yo(PPi) no puede pertenecer a su propio mapa y los que no tienen informacion
                continue

            # posible vecino
            other_x = frame[otherpedindex, 1]
            other_y = frame[otherpedindex, 2]

            if(i==0):
                #calcular_vel_vec(i, sequence_veci,otherpedindex, limites)
                other_x_pos = frame_pos[otherpedindex,1]
                other_y_pos = frame_pos[otherpedindex,2]

                if(other_x_pos==0.0 and other_y_pos==0.0):
                    vel_other=[0,0]  
                else:
                    vel_other = [ other_x_pos-other_x, other_y_pos-other_y]
               
            else:
                other_x_ant = frame_ant[otherpedindex,1]
                other_y_ant = frame_ant[otherpedindex,2]
              
                if(other_x_ant==0.0 and other_y_ant==0.0):
                    vel_other=[0,0]
                else:
                    vel_other = [other_x-other_x_ant, other_y-other_y_ant]
               
            # vectores de los posibles vecinos con su respectiva velocidad 
            vel_p_veci.append(vel_other)
            p_veci.append([other_x,other_y])
        #print(p_veci)
        sequence_vec_flujo[i,:] = llenado_vector_flujo(p_veci, vel_p_veci, obs_traj[i], direcciones[i],ind, i)
    return sequence_vec_flujo

def batch_vecinos(batch_vec, idx, obs_traj, args):
    """
     Recibe: batch_vec tensor de forma [t, obs_len, mnp, 3]
             idx vector de longitud t
             obs_traj es una matriz de la forma [t, obs_len, 2]
     Retorna:
             Una matriz con el vector de flujo optico [t, obs_len, 64]
    """
    t = len(batch_vec)

    vec_flujo = np.zeros((t,args.obs_len,64))

    for ind, sub_batch_vec in enumerate(batch_vec):
    #for ind in range(1):
        person_id = idx[ind]
        
        #sub_batch_vec [obs_len, mnp, 3], las ultimas tres coordendas es id_person, x, y
        #sub_batch_vec= batch_vec[ind]
        
        vec_flujo[ind,:,:] = vector_secuencia_vecinos(sub_batch_vec, person_id, obs_traj[ind],ind) 
    return vec_flujo
