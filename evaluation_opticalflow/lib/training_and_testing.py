import tensorflow as tf
import numpy as np
import math
from tqdm import tqdm

# Parameters
# The only datasets that can use add_social are those of ETH/UCY
# The only datasets that can use add_kp are PETS2009-S2L1, TOWN-CENTRE
class Experiment_Parameters:
    def __init__(self,add_social=False,add_kp=False,obstacles=False):
        # Maximum number of persons in a frame
        self.person_max =70
        # Observation length (trajlet size)
        self.obs_len    = 8
        # Prediction length
        self.pred_len   = 12
        # Flag to consider social interactions
        self.add_social = add_social
        # Number of key points
        self.kp_num     = 18
        # Key point flag
        self.add_kp     = add_kp
        # Obstacles flag
        self.obstacles    = obstacles
        self.intersection = False
        self.delim        = ','
        self.output_representation = 'dxdy' #



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
