import matplotlib.pyplot as plt
import numpy as np
import math

def norm_angle(angle):
    if angle<-math.pi/2.0:
        angle = angle+2.0*math.pi
    if angle>3*math.pi/2.0:
        angle=angle-2.0*math.pi
    return angle

def vector_normal(vec_dir):
    """
    Computes the normal vector.
    Input: a vector (v_x,v_y)
    Output: a vector normal to the input vector. Rotated from the original one by -pi/2
    """
    return np.array([vec_dir[1],-vec_dir[0]],dtype='float')

def vectores_direccion(data):
    """
     Esta funcion recibe una matriz [obs_len,2]
     el cual es la trayectoria del observador
     Input: La matriz data [obs_len, 2]
     Retorna:A [obs_len, 2] matrix of local directions (difference to the anterior)
    """
    pasos = len(data)
    direcciones = np.zeros((pasos,2),dtype='float')

    i=1
    # calculamos la primera direccion la i=1
    if((data[0][0]!=data[1][0]) or (data[0][1]!=data[1][1]) ):
    	direcciones[i] = np.array([data[1][0]-(data[0][0]), data[1][1]-(data[0][1])])
    else:
        # encuentro las primeras dos posiciones diferentes para asi poder calcular el vector direccion
        for j in range(i,pasos-1):
        	if( (data[j][0]!=data[j+1][0]) or (data[j][1]!=data[j+1][1]) ):
        		direcciones[i] = np.array([data[j+1][0]-(data[j][0]), data[j+1][1]-(data[j][1])])
        		break

    direcciones[0] = direcciones[1]

    for i in range(2,pasos,1):
        if( (data[i][0]!=data[i-1][0]) or (data[i][1]!=data[i-1][1])):
            direcciones[i] = np.array([data[i][0]-(data[i-1][0]), data[i][1]-(data[i-1][1])])
        else:
            direcciones[i] = direcciones[i-1]

    return direcciones

# Main class for computing the optical flow
class OpticalFlowSimulator(object):
    def __init__(self, theta0 = 5*np.pi/180.0, thetaf = 175*np.pi/180.0, num_rays=64):
        self.theta0    = theta0
        self.thetaf    = thetaf
        self.delta     = (thetaf-theta0)/num_rays
        self.num_rays  = num_rays
        self.epsilon   = 0.05

    def plot_flow(self,trajectory,neighbors_trajectory,optical_flow,visible_neighbors,visible_obstacles,obstacles,title):
        """ Funcion para graficar y visualizar los vectores y puntos"""
        plt.subplots(4,3,figsize=(15,15))
        for seq_pos in range(1,7):
            plt.subplot(4,3,seq_pos+3*((seq_pos-1)//3))
            current_position  = trajectory[seq_pos]
            current_direction = (trajectory[seq_pos]-trajectory[seq_pos-1])/np.linalg.norm(0.00001+trajectory[seq_pos]-trajectory[seq_pos-1])
            # Whole trajectory
            plt.plot(trajectory[:,0],trajectory[:,1],linewidth=1,color='black')
            # Direction and normal
            vec_norm = vector_normal(current_direction)
            plt.arrow(current_position[0],current_position[1],current_direction[0],current_direction[1],color='blue',linewidth=2)
            plt.arrow(current_position[0],current_position[1],-vec_norm[0],-vec_norm[1],color='red',linewidth=2)
            plt.arrow(current_position[0],current_position[1],+vec_norm[0],+vec_norm[1],color='red',linewidth=2)
            # Plot the neighbors
            for neighbor in neighbors_trajectory[seq_pos]:
                if neighbor[0]>0:
                    plt.plot(neighbor[1],neighbor[2],color='green',marker='o',markersize=14)
                    for neighbor_prev in neighbors_trajectory[seq_pos-1]:
                        if neighbor_prev[0]==neighbor[0]:
                            plt.arrow(neighbor[1],neighbor[2],neighbor[1]-neighbor_prev[1],neighbor[2]-neighbor_prev[2],color='green')
            # Plot the observer agent
            plt.plot(current_position[0],current_position[1],color='blue',marker='o',markersize=14)
            # Plot the visible neighbors
            for neighbor in visible_neighbors[seq_pos]:
                plt.plot(neighbor[0],neighbor[1],color='red',marker='o',markersize=8)
            if visible_obstacles is not None:
                # Plot the visible obstacles
                for vobs in visible_obstacles[seq_pos]:
                    plt.plot(vobs[0],vobs[1],color='magenta',marker='o',markersize=8)
            if obstacles is not None:
                # Draw obstacles
                for obst in obstacles:
                    plt.plot(obst[:,0],obst[:,1],"g-")
            plt.axis('equal')
            # Plot the optical flow
            plt.subplot(4,3,seq_pos+3*(1+(seq_pos-1)//3))
            thetas = np.linspace(self.theta0,self.thetaf,65)[:-1]
            plt.bar(thetas,optical_flow[seq_pos],width=0.05,color='blue')
            plt.plot(thetas,np.zeros_like(thetas))
            plt.xlim((0,3.14))
            plt.ylim((-1,1))
        plt.suptitle(title)
        plt.savefig('./of-sample.pdf')
        plt.show()

    """
    Receives:
            current_position: current position of the person of interest
            neighbor_position: position (x2) of the neighbor
            current_direction: current direction of the person of interest
            neighbors_velocity: velocity (x2) of all the neighbor
    Returns:
        The horizontal component of the optical flow generated by the neighbor (dim. 1)
    """
    def optical_flow_contribution(self,current_position, neighbor_position , current_direction, current_vel, neighbor_velocity, rotation_matrix):
        """ Function that computes the optical flow generated by a neighbor con respecto a PPi"""
        # Relative position (world frame)
        relative_position_w = [neighbor_position[0]-current_position[0],neighbor_position[1]-current_position[1]]

        # Angle corresponding to the current direction
        #theta = math.atan2(current_direction[1], current_direction[0])
        # Rotation matrix
        #mr = np.array( [[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])

        # Transforms the relative position of the neighbor in the rotated frame
        relative_position_l = np.array( [rotation_matrix[0,0]*relative_position_w[0]+rotation_matrix[0,1]*relative_position_w[1],rotation_matrix[1,0]*relative_position_w[0]+rotation_matrix[1,1]*relative_position_w[1]])

        # Process in the same way to express the relative velocity in the rotated frame
        current_vel = np.array(current_vel)
        delta_vel   = neighbor_velocity-current_vel
        delta_vel_l = np.array([rotation_matrix[0,0]*delta_vel[0]+rotation_matrix[0,1]*delta_vel[1],rotation_matrix[1,0]*delta_vel[0]+rotation_matrix[1,1]*delta_vel[1]])

        x = relative_position_l[1]/relative_position_l[0]
        return (delta_vel_l[1]-x*delta_vel_l[0])/relative_position_l[0]

    """
         Receives:
                current_position: current position of the person of interest
                current_direction: current direction of the person of interest
                neighbors_positions: list of positions (x2) of all the neighbors
                neighbors_velocities: list of velocities (x2) of all the neighbors
        Returns:
                An optical flow vector, a list of visible neighbors
    """
    def get_flow_in_cone(self, current_position, current_direction, current_vel, neighbors_positions, neighbors_velocities, obstacles):
        nlen = self.num_rays
        # Closest distances
        closest_squared_distances = np.Inf*np.ones(nlen)
        # Output: the optical flow and the visible neighbors
        flow = np.zeros(nlen)
        tam  = len(neighbors_positions)
        visible_neighbors = np.Inf*np.ones((nlen,2))

        # Compute the Rotation matrix of current position
        # Angle corresponding to the current direction in current position
        theta = math.atan2(current_direction[1], current_direction[0])
        c, s = math.cos(theta), math.sin(theta)
        mr = np.array([[c,-s],[s,c]])

        # Scan the neighbors
        for neighbor_position,neighbor_velocity in zip(neighbors_positions,neighbors_velocities):
            bearing = norm_angle((math.atan2(neighbor_position[1]-current_position[1],neighbor_position[0]-current_position[0])-math.atan2(current_direction[1],current_direction[0])))
            k = int(nlen/2.-bearing/self.delta)
            if k>=0 and k<self.num_rays:
                d =(current_position[0]-neighbor_position[0])**2+(current_position[1]-neighbor_position[1])**2
                # Distance to this neighbor
                if(d<closest_squared_distances[k]):
                    u = self.optical_flow_contribution(current_position,neighbor_position,current_direction,current_vel,neighbor_velocity,mr)
                    flow[k]                      = u
                    closest_squared_distances[k] = d
                    visible_neighbors[k,:]       = neighbor_position
                    # Update JBH: To make the vector less sparse, I duplicate the entry in the neighboring cell
                    if k<self.num_rays-1 and d<closest_squared_distances[k+1]:
                        flow[k+1]                      = u
                        closest_squared_distances[k+1] = d
                        visible_neighbors[k+1,:]       = neighbor_position

        # Test for ray casting: first check if some polygons do intersect the ray.
        visible_obstacles = None
        if obstacles is not None:
            visible_obstacles = np.Inf*np.ones((nlen,2))
            for o,obst in enumerate(obstacles):
                for i in np.arange(0,obst.shape[0]):
                    p1       = obst[i,:]
                    p2       = obst[(i+1)%obst.shape[0],:]
                    dx       = p2[0]-p1[0]
                    dy       = p2[1]-p1[1]
                    bearingm = norm_angle((math.atan2(p1[1]-current_position[1],p1[0]-current_position[0])-math.atan2(current_direction[1],current_direction[0])))
                    bearingp = norm_angle((math.atan2(p2[1]-current_position[1],p2[0]-current_position[0])-math.atan2(current_direction[1],current_direction[0])))
                    km       = int(nlen/2.-bearingm/self.delta)
                    kp       = int(nlen/2.-bearingp/self.delta)
                    # One of the two ends should be visible
                    if (km>=0 and km<self.num_rays) or (kp>=0 and kp<self.num_rays):
                        if bearingm>math.pi/2:
                            if bearingm-bearingp>math.pi:
                                km = self.num_rays
                            else:
                                km = 0
                        if bearingp>math.pi/2:
                            if bearingp-bearingm>math.pi:
                                kp = self.num_rays
                            else:
                                kp = 0
                        # For all the rays in the range
                        for k in range(min(km,kp),max(km,kp)):
                            if k>=0 and k<self.num_rays:
                                # Compute intersection with line
                                sk= np.sin(theta-self.delta*(k-nlen/2))
                                ck= np.cos(theta-self.delta*(k-nlen/2))
                                A = np.array([[ck,-dx],[sk,-dy]])
                                B = np.array([p1[0]-current_position[0],p1[1]-current_position[1]])
                                l = np.linalg.solve(A, B)
                                if l[0]>=0.0 and l[1]>=-self.epsilon and l[1]<=1.0+self.epsilon and l[0]*l[0]<closest_squared_distances[k]:
                                    closest_squared_distances[k] = l[0]*l[0]
                                    visible_obstacles[k,:]       = [current_position[0]+l[0]*ck,current_position[1]+l[0]*sk]
                                    visible_neighbors[k,:]       = [np.Inf,np.Inf]
                                    flow[k]                      = self.optical_flow_contribution(current_position,visible_obstacles[k,:],current_direction,current_vel,[0,0],mr)

        return flow,visible_neighbors,visible_obstacles

    """
        Receives:
            Id: Id of the main agent
            obs_traj is a tensor of shape [t, obs_len, 2] (trajectories)
            neighbors: tensor of shape [obs_len, mnp, 3] (sequence of positions of all the neighbor
        Returns:
            The tensor of optical flow values [t, obs_len, 64]
    """
    def compute_opticalflow_seq(self,Id,obs_traj,neighbors,obstacles):

        direcciones = vectores_direccion(obs_traj)

        # Sequence length
        sequence_length  = neighbors.shape[0]
        # Maximum number of neigbors
        mnp              = neighbors.shape[1]

        # Output
        optical_flow     = np.zeros((sequence_length,self.num_rays), dtype='float')
        visible_neighbors= np.zeros((sequence_length,self.num_rays,2), dtype='float')
        if obstacles is not None:
            visible_obstacles= np.zeros((sequence_length,self.num_rays,2), dtype='float')
        else:
            visible_obstacles= None
        frame_pos = neighbors[1,:,:]
        x_after, y_after = obs_traj[1][0], obs_traj[1][1]
        v_obser = [x_after-obs_traj[0][0],y_after-obs_traj[0][1]]
        # Only used when we work with all neigbors
        vel_before_neighbors = np.zeros((mnp,2), dtype='float')

        # Scan the sequence along time
        for i in range(sequence_length):
            # Current neighbors frame is [mnp,3]
            frame = neighbors[i,:,:]
            # Extract the trajectory of interest (Id)
            person_sec = frame[frame[:,0]== Id,:][0]
            # Current position in the trajectory of interest
            x_current, y_current = obs_traj[i][0], obs_traj[i][1]

            if(i!=0):
                #This is for compute the velocity of the neighbor
                frame_ant = neighbors[i-1,:,:]
                #This is for compute the velocity of the observer agent
                x_after, y_after = obs_traj[i-1][0], obs_traj[i-1][1]
                v_obser = [x_current-x_after, y_current-y_after]

            # List of neighbors
            p_veci     = []
            vel_p_veci = []
            # Scan the other pedestrians

            for other_ped_index in range(mnp):

                # If the neighbor is id==Id or is a no-neighnor (id==0)
                if((frame[other_ped_index, 0] == person_sec[0]) or frame[other_ped_index, 0]==0 ):
                    continue

                # Position of the possible neighbor
                other_x = frame[other_ped_index, 1]
                other_y = frame[other_ped_index, 2]

                # Beginning of the trajectory
                if(i==0):
                    # TODO: to simplify: this velocity is the same as the one at i=1, then we could just copy it after the loop
                    if(frame_pos[other_ped_index,0]==0):
                        vel_other = [0,0]
                        vel_before_neighbors[other_ped_index,:]= vel_other
                    else:
                        other_x_pos = frame_pos[other_ped_index,1]
                        other_y_pos = frame_pos[other_ped_index,2]

                        vel_other = [other_x_pos-other_x,other_y_pos-other_y]
                        vel_before_neighbors[other_ped_index,:]= vel_other
                # Inside the sequence
                else:
                    if(frame_ant[other_ped_index,0]==0):
                        vel_other = vel_before_neighbors[other_ped_index,:]
                        vel_before_neighbors[other_ped_index,:] = vel_other
                    else:
                        other_x_ant = frame_ant[other_ped_index,1]
                        other_y_ant = frame_ant[other_ped_index,2]

                        vel_other = [other_x-other_x_ant, other_y-other_y_ant]
                        vel_before_neighbors[other_ped_index,:]= vel_other
                # Keep the set of velocities
                vel_p_veci.append(vel_other)
                # Keep the set of positions
                p_veci.append([other_x,other_y])

            # Evaluate the flow from the sets of neigbors
            if obstacles is not None:
                optical_flow[i,:], visible_neighbors[i,:,:], visible_obstacles[i,:,:] = self.get_flow_in_cone(obs_traj[i],direcciones[i],v_obser,p_veci,vel_p_veci,obstacles)
            else:
                optical_flow[i,:], visible_neighbors[i,:,:], __ = self.get_flow_in_cone(obs_traj[i],direcciones[i],v_obser,p_veci,vel_p_veci,obstacles)
        return optical_flow, visible_neighbors, visible_obstacles

    # Main function for optical flow computation
    def compute_opticalflow_batch(self,neighbors_batch, idx, obs_traj, obs_len, obstacles):
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
        vec_flow  = np.zeros((t,obs_len,self.num_rays))
        vis_neigh = np.zeros((t,obs_len,self.num_rays,2))
        if obstacles is None:
            vis_obst  = None
        else:
            vis_obst  = np.zeros((t,obs_len,self.num_rays,2))
        for batch_idx, neighbors_descriptor in enumerate(neighbors_batch):
            # Person id
            person_id = idx[batch_idx]
            # Compute the optical flow along this trajectory, given the positions of the neighbors
            if obstacles is not None:
                vec_flow[batch_idx,:,:],vis_neigh[batch_idx,:,:,:],__ =  self.compute_opticalflow_seq(person_id, obs_traj[batch_idx],neighbors_descriptor,obstacles)
            else:
                vec_flow[batch_idx,:,:],vis_neigh[batch_idx,:,:,:],vis_obst[batch_idx,:,:,:] =  self.compute_opticalflow_seq(person_id, obs_traj[batch_idx],neighbors_descriptor,obstacles)
        return vec_flow,vis_neigh,vis_obst
