# Imports
import sys,os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
print('[INF] Tensorflow version: ',tf.__version__)
tf.test.gpu_device_name()
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

from path_prediction.interaction_optical_flow import OpticalFlowSimulator
from path_prediction.process_file import prepare_data
from path_prediction.training_utils import Experiment_Parameters
import random
from datetime import datetime
random.seed(datetime.now())
# To test obstacle-related functions
from path_prediction.obstacles import image_to_world_xy,generate_obstacle_polygons,load_world_obstacle_polygons


# Load the default parameters
experiment_parameters = Experiment_Parameters(add_social=True,add_kp=False,obstacles=True)
# Dataset to be tested
dataset_dir   = "datasets/"
dataset_names = ['eth-hotel','eth-univ','ucy-zara01','ucy-zara02','ucy-univ']
dataset_id    = 1
dataset_name  = dataset_names[dataset_id]
# Load the saved obstacles
obstacles_world = load_world_obstacle_polygons(dataset_dir,dataset_name)
# Process data to get the trajectories
data = prepare_data(dataset_dir, [dataset_name], experiment_parameters)
# Select a random sequence
idSample = random.sample(range(1,data["obs_traj"].shape[0]), 1)
# The random sequence selected
traj_sample             = data['obs_traj'][idSample][0]
traj_neighbors          = data['obs_neighbors'][idSample][0]
optical_flow_sample     = data['obs_optical_flow'][idSample][0]
visible_neighbors_sample= data['obs_visible_neighbors'][idSample][0]
if experiment_parameters.obstacles:
    visible_obst_sample     = data['obs_visible_obstacles'][idSample][0]
else:
    visible_obst_sample     = None
OFSimulator          = OpticalFlowSimulator()

# Plot
OFSimulator.plot_flow(traj_sample,traj_neighbors,optical_flow_sample,visible_neighbors_sample,visible_obst_sample,obstacles_world,title="Sample optical flow, with obstacles")
