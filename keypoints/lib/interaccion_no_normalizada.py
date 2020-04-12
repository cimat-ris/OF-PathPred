import matplotlib.pyplot as plt
import numpy as np
import math

def vector_normal(vec_dir):
    """
    Computes the normal vector.
    Input: a vector (v_x,v_y)
    Output: a vector normal to the input vector. Rotated from the original one by -pi/2
    """
    return np.array([vec_dir[1],-vec_dir[0]])

def vectores_direccion(data):
    """
     Esta funcion recibe una matriz [obs_len,2]
     el cual es la trayectoria del observador
     Input: La matriz data [obs_len, 2]
     Retorna:A [obs_len, 2] matrix of local directions (difference to the anterior)
    """
    pasos = len(data)
    direcciones = np.zeros((pasos,2),dtype='float')

    for i in range(pasos):

        if(i==0 or i==1):
            if((data[0][0]!=data[1][0]) or (data[0][1]!=data[1][1]) ):
                direcciones[i] = np.array([data[1][0]-(data[0][0]), data[1][1]-(data[0][1])])
            else:
                # TODO: Slow and I do not understand well what it does
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

def in_line(Point_eval, vec_dir, Point_on_line):
    """ Function to evaluate where a point is with respect to a straight line"""
    # Positive: right side of the line
    # Negative: left side of the line
    # Zero: on the line
    return np.sum(Point_eval*vector_normal(vec_dir)) - np.sum(Point_on_line*vector_normal(vec_dir))

def in_cone(Point_eval,vec_dir1,vec_dir2,Point_in_recta):
    """ FuFunction to evaluate where a point is inside or outside of a cone"""
    # 1 if inside thecone
    # 0 if outside
    # We evaluate the relative position with the two lines forming the cone
    if in_line(Point_eval,vec_dir1,Point_in_recta)>0 and in_line(Point_eval,vec_dir2,Point_in_recta)<0:
        return 1
    return 0

def grafica_vec(ind,i,PPi,vec_dir,list_vect_new=[],Points=[],factor=10000):
    """ Funcion para graficar y visualizar los vectores y puntos"""

    # Graficamos el vector direccion y el normal
    vec_norm = vector_normal(vec_dir)
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


"""
     Receives:
            current_direction: current direction of the person of interest
            theta0: initial value of the angle (in radians)
            thetaf: final value of the angle (in radians)
            num_rays: number of rays
     Returns:
            A vector of directions partitioning the cone
"""
def get_partition_cone(current_direction,theta0,thetaf,num_rays):
    """ Function to partition the cone """

    # Normal to the current orientation: rotated from the direction by -PI/2
    vec_norm = vector_normal(current_direction)
    list_vect_new = []
    # Norm of the direction vector
    norm = np.linalg.norm(current_direction)
    # Fills the vector above with
    for theta_i in np.linspace(theta0,thetaf,num_rays+1):
        # Rotated vector
        vec_new =   norm*(np.sin(theta_i)*current_direction - np.cos(theta_i)*vec_norm)
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
            if in_cone(coord,list_vect_new[k],list_vect_new[k+1],PPi):

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

"""
     Receives:
            neighbors_positions: list of positions (x2) of all the neighbors
            neighbors_velocities: list of velocities (x2) of all the neighbors
            current_position: current position of the person of interest
            current_direction: current direction of the person of interest
     Returns:
            The optical flow vector (dim. 64)
"""
def fill_optical_flow(neighbors_positions, neighbors_velocities,  current_position, current_direction):
    # TODO: define the parameters as parameters of a class
    # Visibility cone
    theta0   = 5*np.pi/180.0
    thetaf   = 175*np.pi/180.0
    num_part = 64
    # Filter
    list_vect_new = get_partition_cone( current_direction, theta0, thetaf, num_part)
    optical_flow, vecinos = get_vector_cono_flujo(current_position,current_direction,neighbors_positions,neighbors_velocities, list_vect_new)
    return optical_flow
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

"""
     Receives:
            neighbors: tensor of shape [obs_len, mnp, 3] (sequence of positions of all the neighbors)
            idx vector of length t (batches of ids of each trajectory) [t]
            obs_traj is a tensor of shape [t, obs_len, 2] (trajectories)
     Returns:
            The tensor of optical flow values [t, obs_len, 64]
"""
def compute_opticalflow(neighbors, Id, obs_traj,ind):

    # Direction vectors
    direcciones = vectores_direccion(obs_traj)

    # Sequence length
    sequence_length  = neighbors.shape[0]
    # Maximum number of neigbors
    mnp              = neighbors.shape[1]

    # Output
    optical_flow = np.zeros((sequence_length,64), dtype='float')

    for i in range(sequence_length):

        # Current frame is [mnp,3]
        frame = neighbors[i,:,:]

        # Extract next/previous frames
        if(i==0):
            frame_pos = neighbors[i+1,:,:]
        else:
            frame_ant = neighbors[i-1,:,:]

        # Extract the trajectory of interest (Id)
        person_sec = frame[frame[:,0]== Id,:][0]
        # Current position in the trajectory of interest
        x_current, y_current = obs_traj[i][0], obs_traj[i][1]

        p_veci     = []
        vel_p_veci = []

        # Scan the other pedestrians
        for other_ped_index in range(mnp):

            # modifique
            if((frame[other_ped_index, 0] == person_sec[0]) or frame[other_ped_index, 0]==0 ):
                # Yo(PPi) no puede pertenecer a su propio mapa y los que no tienen informacion
                continue

            # Position of th possible neighbor
            other_x = frame[other_ped_index, 1]
            other_y = frame[other_ped_index, 2]

            # Beginning of the trajectory
            if(i==0):
                other_x_pos = frame_pos[other_ped_index,1]
                other_y_pos = frame_pos[other_ped_index,2]
                # TODO: to simplify: this velocity is the same as the one at i=1, then we could just copy it after the loop
                if(other_x_pos==0.0 and other_y_pos==0.0):
                    vel_other=[0,0]
                else:
                    vel_other = [ other_x_pos-other_x, other_y_pos-other_y]
            # Inside the sequence
            else:
                other_x_ant = frame_ant[other_ped_index,1]
                other_y_ant = frame_ant[other_ped_index,2]
                if(other_x_ant==0.0 and other_y_ant==0.0):
                    vel_other=[0,0]
                else:
                    vel_other = [other_x-other_x_ant, other_y-other_y_ant]

            # Keep the set of velocities
            vel_p_veci.append(vel_other)
            # Keep the set of positions
            p_veci.append([other_x,other_y])
        # Evaluate the flow from the sets of neigbors
        optical_flow[i,:] = fill_optical_flow(p_veci, vel_p_veci, obs_traj[i], direcciones[i])
    return optical_flow

# Main function for optical flow computation
def compute_opticalflow_batch(neighbors_batch, idx, obs_traj, obs_len):
    """
     Receives:
            neighbors_batch: tensor of shape [t, obs_len, mnp, 3] (batch of positions of all the neighbors)
            idx vector of length t (batches of ids of each trajectory) [t]
            obs_traj is a tensor of shape [t, obs_len, 2] (trajectories)
     Returns:
            The tensor of optical flow values [t, obs_len, 64]
    """
    # Length of the batch
    t = len(neighbors_batch)
    # Scan the neighbors_batch tensor
    vec_flow = np.zeros((t,obs_len,64))
    for batch_idx, neighbors_descriptor in enumerate(neighbors_batch):
        # Person id
        person_id = idx[batch_idx]
        # Compute the optical flow along this trajectory, given the positions of the neighbors
        vec_flow[batch_idx,:,:] =  compute_opticalflow(neighbors_descriptor, person_id, obs_traj[batch_idx],batch_idx)
    return vec_flow
