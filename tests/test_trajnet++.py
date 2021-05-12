import sys,os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import argparse
import time
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
print('[INF] Tensorflow version: ',tf.__version__)
tf.test.gpu_device_name()
import random
from datetime import datetime
random.seed(datetime.now())
import matplotlib.pyplot as plt

# Important imports
import path_prediction.batches_data
from path_prediction.training_utils import Experiment_Parameters
from path_prediction.interaction_optical_flow import OpticalFlowSimulator
from path_prediction.process_file import prepare_data_trajnetplusplus
from path_prediction.datasets_utils import setup_trajnetplusplus_experiment
import logging
import socket
import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs_length', default=8, type=int,help='observation length')
    parser.add_argument('--pred_length', default=12, type=int,help='prediction length')
    parser.add_argument('--path', default='trajdata',help='glob expression for data files')
    args     = parser.parse_args()
    train_dataset_names = ["biwi_hotel","crowds_students001","crowds_students003","crowds_zara01","crowds_zara03","lcas","wildtrack","cff_06","cff_07","cff_08"]
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
