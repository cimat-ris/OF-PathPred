import sys, os, argparse, logging,random, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import tensorflow as tf
from datetime import datetime
random.seed(datetime.now())
import matplotlib.pyplot as plt

# Important imports
import path_prediction.batches_data
from path_prediction.training_utils import Experiment_Parameters
from path_prediction.interaction_optical_flow import OpticalFlowSimulator
from path_prediction.process_file import prepare_data_trajnetplusplus
from path_prediction.datasets_utils import setup_trajnetplusplus_experiment
import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs_length', default=8, type=int,help='observation length')
    parser.add_argument('--pred_length', default=12, type=int,help='prediction length')
    parser.add_argument('--path', default='datasets/trajnetplusplus',help='glob expression for data files')
    parser.add_argument('--log_level',type=int, default=20,help='Log level (default: 20)')
    parser.add_argument('--log_file',default='',help='Log file (default: standard output)')
    args     = parser.parse_args()
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


    #train_dataset_names = ["biwi_hotel","crowds_students001","crowds_students003","crowds_zara01","crowds_zara03","lcas","wildtrack","cff_06","cff_07","cff_08"]
    train_dataset_names = ["biwi_hotel","crowds_students001","crowds_students003","crowds_zara01","crowds_zara03","lcas","wildtrack","cff_08"]
    test_dataset_names = ["biwi_eth"]

    # Load the default parameters
    experiment_parameters = Experiment_Parameters(add_social=True,add_kp=False)
    experiment_parameters.obs_len  = args.obs_length
    experiment_parameters.pred_len = args.pred_length
    # Load the datasets
    train_data,val_data,test_data = setup_trajnetplusplus_experiment('TRAJNETPLUSPLUS',args.path,train_dataset_names,test_dataset_names,experiment_parameters)
    print("[INF] Total number of training trajectories:",train_data["obs_traj"].shape[0])
    # Training dataset
    pickle_out = open('test.pickle',"wb")
    pickle.dump(train_data, pickle_out, protocol=2)
    pickle_out.close()
    # Select a random sequence within this dataset
    idSample = random.sample(range(1,test_data["obs_traj"].shape[0]), 1)
    # The random sequence selected
    traj_sample             = test_data['obs_traj'][idSample][0]
    traj_neighbors          = test_data['obs_neighbors'][idSample][0]
    optical_flow_sample     = test_data['obs_optical_flow'][idSample][0]
    visible_neighbors_sample= test_data['obs_visible_neighbors'][idSample][0]
    OFSimulator          = OpticalFlowSimulator()
    # Plot simulated optical flow
    OFSimulator.plot_flow(traj_sample,traj_neighbors,optical_flow_sample,visible_neighbors_sample,None,None,title="Sample optical flow")

if __name__ == '__main__':
    main()
