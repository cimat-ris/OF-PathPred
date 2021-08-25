# Imports
import sys, os, argparse, logging, random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from path_prediction.utils.interaction_optical_flow import OpticalFlowSimulator
from path_prediction.utils.process_file import prepare_data
from path_prediction.utils.training_utils import Experiment_Parameters
from datetime import datetime
random.seed()
# To test obstacle-related functions
from path_prediction.utils.obstacles import image_to_world_xy,generate_obstacle_polygons,load_world_obstacle_polygons


def main():
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='datasets/',
                        help='glob expression for data files')
    parser.add_argument('--obstacles', dest='obstacles', action='store_true',help='includes the obstacles in the optical flow')
    parser.set_defaults(obstacles=False)
    parser.add_argument('--log_level',type=int, default=20,help='Log level (default: 20)')
    parser.add_argument('--log_file',default='',help='Log file (default: standard output)')
    parser.add_argument('--dataset_id', '--id',
                    type=int, default=0,help='dataset id (default: 0)')
    parser.add_argument('--samples',type=int, default=5,help='dataset id (default: 0)')
    args = parser.parse_args()
    if args.log_file=='':
        logging.basicConfig(format='%(levelname)s: %(message)s',level=args.log_level)
    else:
        logging.basicConfig(filename=args.log_file,format='%(levelname)s: %(message)s',level=args.log_level)
    # Info about TF and GPU
    logging.info('Tensorflow version: {}'.format(tf.__version__))
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices)>0:
        logging.info('Using GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        logging.info('Using CPU')

    # Load the default parameters
    experiment_parameters =     Experiment_Parameters(obstacles=args.obstacles)
    # Dataset to be tested
    dataset_dir   = args.path
    dataset_names = ['eth-hotel','eth-univ','ucy-zara01','ucy-zara02','ucy-univ']
    dataset_id    = args.dataset_id
    dataset_name  = dataset_names[dataset_id]
    # Load the saved obstacles
    obstacles_world = load_world_obstacle_polygons(dataset_dir,dataset_name)
    # Process data to get the trajectories (just for dataset_id)
    data = prepare_data(dataset_dir, [dataset_name], experiment_parameters)

    for i in range(args.samples):
        # Select a random sequence within this dataset
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
            # Optical flow
        OFSimulator          = OpticalFlowSimulator(log_polar_mapping=False)
        # Plot simulated optical flow
        OFSimulator.plot_flow(traj_sample,traj_neighbors,optical_flow_sample,visible_neighbors_sample,visible_obst_sample,obstacles_world,title="Sample optical flow")


if __name__ == '__main__':
    main()
