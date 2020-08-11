# Imports
import sys,os
sys.path.append('../lib')
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
print('Tensorflow version: ',tf.__version__)
tf.test.gpu_device_name()

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

from interaction_optical_flow import OpticalFlowSimulator
from process_file import process_file, predictor_parameters
import random
from datetime import datetime
random.seed(datetime.now())
# To test obstacle-related functions
from obstacles import image_to_world_xy,generate_obstacle_polygons,load_world_obstacle_polygons
import matplotlib.pyplot as plt

# Dataset to be tested
dataset_paths  = "../../data1/"
#dataset_name = 'eth-hotel'
#dataset_name = 'eth-univ'
#dataset_name = 'ucy-zara01'
dataset_name = 'ucy-zara02'

# File of trajectories coordinates. Coordinates are in world frame
data_path = dataset_paths+dataset_name
# Load the saved obstacles
obstacles_world = load_world_obstacle_polygons(dataset_paths,dataset_name)
# Load the default parameters
parameters = predictor_parameters(add_social=True)
# Process data to get the trajectories
data = process_file(data_path, parameters, ',')
# Select a random sequence
idSample = random.sample(range(1,data["obs_traj"].shape[0]), 1)
# The random sequence selected
traj_sample   = data['obs_traj'][idSample][0]
traj_neighbors= data['obs_neighbors'][idSample][0]
traj_id       = data['key_idx'][idSample]
OFSimulator          = OpticalFlowSimulator()

# Optical flow: no obstacle
optical_flow_sample,visible_neighbors_sample,visible_obst_sample = OFSimulator.compute_opticalflow_seq(traj_id,traj_sample,traj_neighbors,obstacles_world)
# Plot
OFSimulator.plot_flow(traj_sample,traj_neighbors,optical_flow_sample,visible_neighbors_sample,visible_obst_sample,obstacles_world,title="Sample optical flow, with obstacles")

# Optical flow: with obstacles
optical_flow_sample,visible_neighbors_sample,visible_obst_sample = OFSimulator.compute_opticalflow_seq(traj_id,traj_sample,traj_neighbors,None)
# Plot
OFSimulator.plot_flow(traj_sample,traj_neighbors,optical_flow_sample,visible_neighbors_sample,None,None,title="Sample optical flow, with no obstacle")
