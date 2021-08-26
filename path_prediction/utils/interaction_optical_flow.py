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
    	direcciones[i] = np.array([data[1][0]-(data[0][0]),data[1][1]-(data[0][1])])
    else:
        # encuentro las primeras dos posiciones diferentes para asi poder calcular el vector direccion
        for j in range(i,pasos-1):
        	if( (data[j][0]!=data[j+1][0]) or (data[j][1]!=data[j+1][1]) ):
        		direcciones[i] = np.array([data[j+1][0]-(data[j][0]),data[j+1][1]-(data[j][1])])
        		break

    direcciones[0] = direcciones[1]

    for i in range(2,pasos,1):
        if( (data[i][0]!=data[i-1][0]) or (data[i][1]!=data[i-1][1])):
            direcciones[i] = np.array([data[i][0]-(data[i-1][0]),data[i][1]-(data[i-1][1])])
        else:
            direcciones[i] = direcciones[i-1]

    return direcciones

# Main class for computing the optical flow
class OpticalFlowSimulator(object):
    """
    Model parameters.
    """
    class Parameters():
        def __init__(self,log_polar_mapping=False):
            # -----------------
            self.a         = 1.0
            self.num_rays  = 64
            self.theta0    = 5*np.pi/180.0
            self.thetaf    = 175*np.pi/180.0
            self.delta     = (self.thetaf-self.theta0)/self.num_rays
            self.deltalog  = (np.log(self.a + self.thetaf - np.pi*0.5)-np.log(self.a))/(self.num_rays*0.5)
            self.use_bounds= False
            self.lim       = [0,0,0,0,1]
            self.log_polar_mapping= log_polar_mapping

    def __init__(self, parameters=Parameters()):
        self.parameters  = parameters
        self.epsilon     = 0.05
        width            = self.parameters.lim[1]-self.parameters.lim[0]
        height           = self.parameters.lim[3]-self.parameters.lim[2]
        self.height_bound= height/self.parameters.lim[4]
        self.width_bound = width/self.parameters.lim[4]

    def plot_flow(self,trajectory,neighbors_trajectory,optical_flow,visible_neighbors,visible_obstacles,obstacles,title):
        """ Visualization of the optical flow and current spatial situation"""
        plt.subplots(2,3,figsize=(15,15))
        # Six consecutive timesteps
        for seq_pos in range(1,4):
            # Definition of the subplot
            ax =plt.subplot(2,3,seq_pos+3*((seq_pos-1)//3))
            ax.set_aspect('equal')
            current_position  = trajectory[seq_pos]
            current_direction = trajectory[seq_pos]-trajectory[seq_pos-1]
            # Plot the neighbors
            for neighbor in neighbors_trajectory[seq_pos]:
                if neighbor[0]>0:
                    plt.plot(neighbor[1],neighbor[2],color='green',marker='o',markersize=10)
                    # Neighbors directions
                    for neighbor_prev in neighbors_trajectory[seq_pos-1]:
                        if neighbor_prev[0]==neighbor[0]:
                            neighbor_direction = (neighbor[1:3]-neighbor_prev[1:3])/(0.00001+np.linalg.norm(neighbor[1:3]-neighbor_prev[1:3]))
                            if (not np.isnan(neighbor_prev[:]).any()) and (not np.isnan(neighbor[:]).any()):
                                plt.arrow(neighbor[1],neighbor[2],neighbor_direction[0],neighbor_direction[1],color='green')
            # Plot the visible neighbors
            for neighbor in visible_neighbors[seq_pos]:
                plt.plot(neighbor[0],neighbor[1],color='red',marker='o',markersize=6)
            if self.parameters.use_bounds:
                width_low    = current_position[0]-self.width_bound/2.0
                width_high   = current_position[0]+self.width_bound/2.0
                height_low   = current_position[1]-self.height_bound/2.0
                height_high  = current_position[1]+self.height_bound/2.0
                plt.plot([width_low,width_high],[height_low,height_low], "#ff7700")
                plt.plot([width_low,width_low],[height_low ,height_high], "#ff7700")
                plt.plot([width_high,width_high],[height_low,height_high], "#ff7700")
                plt.plot([width_low,width_high],[height_high,height_high], "#ff7700")

            if visible_obstacles is not None:
                # Plot the visible obstacles
                for vobs in visible_obstacles[seq_pos]:
                    plt.plot(vobs[0],vobs[1],color='magenta',marker='o',markersize=5)
            if obstacles is not None:
                # Draw obstacles
                for obst in obstacles:
                    plt.fill(obst[:,0],obst[:,1],"g-")
            # Plot the agent of interest
            plt.plot(current_position[0],current_position[1],color='blue',marker='o',markersize=10)
            # Direction and normal
            vec_cur  = current_direction/(np.linalg.norm(current_direction)+0.001)
            if np.linalg.norm(current_direction)>0:
                vec_norm = vector_normal(current_direction)
                vec_l    = 2.0
                plt.arrow(current_position[0],current_position[1],vec_l*vec_cur[0],vec_l*vec_cur[1],color='black',linewidth=1)
                plt.arrow(current_position[0],current_position[1],-vec_l*vec_norm[0],-vec_l*vec_norm[1],color='red',linewidth=1)
                plt.arrow(current_position[0],current_position[1],+vec_l*vec_norm[0],+vec_l*vec_norm[1],color='red',linewidth=1)
            else:
                plt.plot(current_position[0],current_position[1],color='white',marker='o',markersize=7)
            # Whole trajectory of the agent of interest
            plt.plot(trajectory[:,0],trajectory[:,1],linewidth=1,color='blue')
            # 20m wide
            plt.xlim((current_position[0]-10.0,current_position[0]+10.0))
            # Plot the optical flow
            plt.subplot(2,3,seq_pos+3*(1+(seq_pos-1)//3))
            thetas = np.linspace(self.parameters.theta0-math.pi*0.5,self.parameters.thetaf-math.pi*0.5,65)[:-1]
            plt.bar(thetas,optical_flow[seq_pos],width=0.05,color='blue')
            plt.plot(thetas,np.zeros_like(thetas))
            plt.xlim((-math.pi*0.5,math.pi*0.5))
            if self.parameters.log_polar_mapping:
                plt.xscale('symlog', linthresh=1.0)
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
            rotation_matrix: 2D rotation from world frame to local frame
    Returns:
        The horizontal component of the optical flow generated by the neighbor (dim. 1)
    """
    def optical_flow_contribution(self,current_position, neighbor_position , current_direction, current_velocity, neighbor_velocity, rotation_matrix):
        """ Function that computes the optical flow generated by a neighbor"""
        # Relative position (worotation_matrixrld frame)
        relative_position_w = neighbor_position[:]-current_position[:]
        # Transforms the relative position of the neighbor in the local frame
        relative_position_l = rotation_matrix @ relative_position_w
        # Process in the same way to express the relative velocity in the rotated frame
        delta_vel   = neighbor_velocity-np.array(current_velocity)
        delta_vel_l = rotation_matrix @ delta_vel
        u           = relative_position_l[0]/relative_position_l[1]
        return (delta_vel_l[0]-u*delta_vel_l[1])/relative_position_l[1]

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
        nlen                      = self.parameters.num_rays
        # Closest distances
        closest_squared_distances = np.Inf*np.ones(nlen)
        # Output: the optical flow and the visible neighbors
        flow                      = np.zeros(nlen)
        visible_neighbors         = np.Inf*np.ones((nlen,2))

        # Compute the Rotation matrix of current position
        # Angle corresponding to the current direction in current position
        theta           = math.atan2(current_direction[1], current_direction[0])
        # Rotation matrix to map from **world** to **local** frame
        c, s            = math.cos(theta-math.pi*0.5), math.sin(theta-math.pi*0.5)
        mr              = np.array([[c,s],[-s,c]])

        # Scan the neighbors
        for neighbor_position,neighbor_velocity in zip(neighbors_positions,neighbors_velocities):
            bearing = norm_angle(math.atan2(neighbor_position[1]-current_position[1],neighbor_position[0]-current_position[0])-theta)
            # Check if it is visible (bearing should be between -pi/2 and pi/2)
            if self.parameters.log_polar_mapping==False:
                k = int(nlen/2.-bearing/self.parameters.delta)
            else:
                if bearing>=0:
                    k = int(nlen/2.-np.log(self.parameters.a+bearing)/self.parameters.deltalog)
                else:
                    k = int(nlen/2.-(2.0*np.log(self.parameters.a)-np.log(self.parameters.a-bearing))/self.parameters.deltalog)
            if k>=0 and k<self.parameters.num_rays:
                if bearing<self.parameters.theta0-math.pi*0.5 or bearing>self.parameters.thetaf+math.pi*0.5:
                    logging.error("This value of bearing should NOT happen {}:".format(bearing))
                d =(current_position[0]-neighbor_position[0])**2+(current_position[1]-neighbor_position[1])**2
                # Distance to this neighbor
                if(d<closest_squared_distances[k]):
                    u = self.optical_flow_contribution(current_position,neighbor_position,current_direction,current_vel,neighbor_velocity,mr)
                    if self.parameters.log_polar_mapping:
                        u = u/(self.parameters.a+np.abs(bearing))
                    flow[k]                      = u
                    closest_squared_distances[k] = d
                    visible_neighbors[k,:]       = neighbor_position
                    # Update JBH: To make the vector less sparse, I duplicate the entry in the neighboring cell
                    if k<self.parameters.num_rays-1 and d<closest_squared_distances[k+1]:
                        flow[k+1]                      = u*0.5
                        closest_squared_distances[k+1] = d
                    if k>0 and d<closest_squared_distances[k-1]:
                        flow[k-1]                      = u*0.5
                        closest_squared_distances[k-1] = d

        # Test for ray casting: first check if some polygons do intersect the ray.
        visible_obstacles = None
        if obstacles is not None:
            visible_obstacles = np.Inf*np.ones((nlen,2))
            for o,obst in enumerate(obstacles):
                for i in np.arange(0,obst.shape[0]):
                    # Consecutive points
                    p1       = obst[i,:]
                    p2       = obst[(i+1)%obst.shape[0],:]
                    dp       = p2[:]-p1[:]
                    bearingm = norm_angle((math.atan2(p1[1]-current_position[1],p1[0]-current_position[0])-theta))
                    bearingp = norm_angle((math.atan2(p2[1]-current_position[1],p2[0]-current_position[0])-theta))
                    km       = int(nlen/2.-bearingm/self.parameters.delta)
                    kp       = int(nlen/2.-bearingp/self.parameters.delta)
                    # One of the two ends should be visible
                    if (km>=0 and km<self.parameters.num_rays) or (kp>=0 and kp<self.parameters.num_rays):
                        if bearingm>math.pi/2:
                            if bearingm-bearingp>math.pi:
                                km = self.parameters.num_rays
                            else:
                                km = 0
                        if bearingp>math.pi/2:
                            if bearingp-bearingm>math.pi:
                                kp = self.parameters.num_rays
                            else:
                                kp = 0
                        # For all the rays in the range
                        for k in range(min(km,kp),max(km,kp)):
                            if k>=0 and k<self.parameters.num_rays:
                                # Compute intersection with line
                                sk= np.sin(theta-self.parameters.delta*(k-nlen/2))
                                ck= np.cos(theta-self.parameters.delta*(k-nlen/2))
                                A = np.array([[ck,-dp[0]],[sk,-dp[1]]])
                                B = np.array([p1[0]-current_position[0],p1[1]-current_position[1]])
                                l = np.linalg.solve(A, B)
                                if l[0]>=0.0 and l[1]>=-self.parameters.epsilon and l[1]<=1.0+self.parameters.epsilon and l[0]*l[0]<closest_squared_distances[k]:
                                    closest_squared_distances[k] = l[0]*l[0]
                                    visible_obstacles[k,:]       = [current_position[0]+l[0]*ck,current_position[1]+l[0]*sk]
                                    visible_neighbors[k,:]       = [np.Inf,np.Inf]
                                    flow[k]                      = self.optical_flow_contribution(current_position,visible_obstacles[k,:],current_direction,current_vel,[0,0],mr)
        return flow,visible_neighbors,visible_obstacles

    """
        Receives:
            obs_traj: [obs_len, 2] tensor (trajectories of interest)
            neighbors: [obs_len, mnp, 3] tensor (sequence of positions of all the neighbors
        Returns:
            The tensor of optical flow values [t, obs_len, 64]
    """
    def compute_opticalflow_seq(self,obs_traj,neighbors,obstacles):

        directions = vectores_direccion(obs_traj)
        # Sequence length
        sequence_length  = neighbors.shape[0]
        # Maximum number of neigbors
        mnp              = neighbors.shape[1]
        # Output
        optical_flow     = np.zeros((sequence_length,self.parameters.num_rays), dtype='float')
        visible_neighbors= np.zeros((sequence_length,self.parameters.num_rays,2), dtype='float')
        if obstacles is not None:
            visible_obstacles= np.zeros((sequence_length,self.parameters.num_rays,2), dtype='float')
        else:
            visible_obstacles = None
        # Only used when we work with all neighbors
        velocity_main         = np.zeros((sequence_length,2), dtype='float')
        velocity_main[1:]     = obs_traj[1:][:]-obs_traj[:-1][:]
        velocity_main[0]      = velocity_main[1]
        velocity_other        = np.zeros((sequence_length,mnp,2), dtype='float')
        velocity_other[1:,:,:]= neighbors[1:,:,0:2]-neighbors[:-1,:,0:2]
        velocity_other[0]     = velocity_other[1]
        # Scan the sequence along time
        for i in range(sequence_length):
            # Current neighbors frame is [mnp,3]
            current_frame = neighbors[i,:,:]
            # List of neighbors
            positions_other = []
            velocities_other= []
            # Scan the other pedestrians
            for other_ped_index in range(mnp):
                if current_frame[other_ped_index,0]==0.0:
                    continue
                # TODO: Would be more efficient with test on Isd
                # If the neighbor is id==Id or is a no-neighbor (id==Nan)
                if(((current_frame[other_ped_index, 1:3]==obs_traj[i][:]).all()) ):
                    continue
                # Check if the neighbor is out of bounds
                if self.parameters.use_bounds and (np.abs(current_frame[other_ped_index, 1:3]-obs_traj[i][:])>self.parameters.width_bound/2.0).any():
                    continue
                # Keep the set of velocities
                velocities_other.append(velocity_other[i,other_ped_index,:])
                # Keep the set of positions
                positions_other.append(current_frame[other_ped_index, 1:3])

            # Evaluate the flow from the sets of neigbors
            if obstacles is not None:
                optical_flow[i,:], visible_neighbors[i,:,:], visible_obstacles[i,:,:] = self.get_flow_in_cone(obs_traj[i],directions[i],velocity_main[i,:],positions_other,velocities_other,obstacles)
            else:
                optical_flow[i,:], visible_neighbors[i,:,:], __ = self.get_flow_in_cone(obs_traj[i],directions[i],velocity_main[i,:],positions_other,velocities_other,obstacles)
        return optical_flow, visible_neighbors, visible_obstacles

    # Main function for optical flow computation
    def compute_opticalflow_batch(self,neighbors_batch, obs_traj_batch, obs_len, obstacles):
        """
        Receives:
                neighbors_batch: tensor of shape [t, obs_len, mnp, 3] (batch of positions of all the neighbors)
                obs_traj is a tensor of shape [t, obs_len, 2] (trajectories)
        Returns:
                The tensor of optical flow values [t, obs_len, num_rays]
        """
        # Length of the batch
        t = len(neighbors_batch)
        vec_flow  = np.zeros((t,obs_len,self.parameters.num_rays))
        vis_neigh = np.zeros((t,obs_len,self.parameters.num_rays,2))
        if obstacles is None:
            vis_obst  = None
        else:
            vis_obst  = np.zeros((t,obs_len,self.parameters.num_rays,2))
        # Scan the neighbors_batch tensor
        for batch_idx, neighbors_descriptor in enumerate(neighbors_batch):
            # Compute the optical flow along this trajectory, given the positions of the neighbors
            if obstacles is None:
                vec_flow[batch_idx,:,:],vis_neigh[batch_idx,:,:,:],__ =  self.compute_opticalflow_seq(obs_traj_batch[batch_idx],neighbors_descriptor,obstacles)
            else:
                vec_flow[batch_idx,:,:],vis_neigh[batch_idx,:,:,:],vis_obst[batch_idx,:,:,:] =  self.compute_opticalflow_seq(obs_traj_batch[batch_idx],neighbors_descriptor,obstacles)
        return vec_flow,vis_neigh,vis_obst
