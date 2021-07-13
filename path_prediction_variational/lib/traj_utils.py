import numpy as np

def relative_to_abs(rel_traj, start_pos):
    """Relative x,y to absolute x,y coordinates.
    Args:
    rel_traj: numpy array [T,2]
    start_pos: [2]
    Returns:
    abs_traj: [T,2]
    """
    # batch, seq_len, 2
    # the relative xy cumulated across time first
    displacement = np.cumsum(rel_traj, axis=0)
    abs_traj = displacement + np.array([start_pos])  # [1,2]
    return abs_traj

def vw_to_abs(vw_traj, start_pos):
    """v,w to absolute x,y coordinates.
    Args:
    vw_traj: numpy array [T,2]
    start_pos: [2]
    start_t: [1]
    Returns:
    abs_traj: [T,2]
    """
    # batch, seq_len, 2
    orientations = 10.0*vw_traj[:,1]
    dx           = np.multiply(np.cos(orientations),0.5*np.exp(vw_traj[:,0]))
    dy           = np.multiply(np.sin(orientations),0.5*np.exp(vw_traj[:,0]))
    abs_traj     = np.zeros_like(vw_traj)
    abs_traj[0,0]=  start_pos[0] + dx[0]
    abs_traj[0,1]=  start_pos[1] + dy[0]
    for i in range(1,vw_traj.shape[0]):
        abs_traj[i,0] =  abs_traj[i-1,0] + dx[i]
        abs_traj[i,1] =  abs_traj[i-1,1] + dy[i]
    return abs_traj
