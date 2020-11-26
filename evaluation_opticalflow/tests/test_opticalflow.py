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
from process_file import process_file
from training_and_testing import Experiment_Parameters
import random
from datetime import datetime
random.seed(datetime.now())
# To test obstacle-related functions
from obstacles import image_to_world_xy,generate_obstacle_polygons,load_world_obstacle_polygons
import matplotlib.pyplot as plt


# Load the default parameters
experiment_parameters = Experiment_Parameters(add_social=True,add_kp=False,obstacles=False)
# Dataset to be tested
dataset_paths  = "../../datasets/"
#dataset_name = 'eth-hotel'
#dataset_name = 'eth-univ'
#dataset_name = 'ucy-zara01'
dataset_name = 'ucy-zara02'
# File of trajectories coordinates. Coordinates are in world frame
data_paths = [dataset_paths+dataset_name]
# Load the saved obstacles
obstacles_world = load_world_obstacle_polygons(dataset_paths,dataset_name)
# Process data to get the trajectories
data = process_file(data_paths, experiment_parameters)
# Select a random sequence
idSample = random.sample(range(1,data["obs_traj"].shape[0]), 1)
# The random sequence selected
traj_sample             = data['obs_traj'][idSample][0]
traj_neighbors          = data['obs_neighbors'][idSample][0]
optical_flow_sample     = data['obs_optical_flow'][idSample][0]
visible_neighbors_sample= data['obs_visible_neighbors'][idSample][0]
if experiment_parameters.obstacles:
    visible_obst_sample     = data['visible_obstacles'][idSample][0]
else:
    visible_obst_sample     = None
OFSimulator          = OpticalFlowSimulator()

# Plot
OFSimulator.plot_flow(traj_sample,traj_neighbors,optical_flow_sample,visible_neighbors_sample,visible_obst_sample,obstacles_world,title="Sample optical flow, with obstacles")

# Optical flow: with obstacles
# Plot
#OFSimulator.plot_flow(traj_sample,traj_neighbors,optical_flow_sample,visible_neighbors_sample,None,None,title="Sample optical flow, with no obstacle")
