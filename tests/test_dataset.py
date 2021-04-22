import sys,os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import argparse
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
print('[INF] Tensorflow version: ',tf.__version__)
tf.test.gpu_device_name()
import random
from datetime import datetime
random.seed(datetime.now())
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

# Important imports
from path_prediction.process_file import process_file
import path_prediction.batches_data
from path_prediction.training_utils import Experiment_Parameters


def main():
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='datasets/',
                        help='glob expression for data files')
    args = parser.parse_args()

    # Load the default parameters
    experiment_parameters = Experiment_Parameters(add_social=False,add_kp=False,obstacles=False)

    # Dataset to be tested
    dataset_dir   = "datasets/"
    dataset_names = ['eth-hotel','eth-univ','ucy-zara01','ucy-zara02','ucy-univ']

    # Process data specified by the path to get the trajectories with
    data = process_file(dataset_dir, dataset_names, experiment_parameters)

    # Muestreamos aleatoriamente para separar datos de entrenamiento, validacion y prueba
    training_pc  = 0.7
    test_pc      = 0.2
    # Count how many data we have (sub-sequences of length 8, in pred_traj)
    ndata      = len(data[list(data.keys())[2]])
    idx        = random.sample(range(ndata), ndata)
    training   = int(ndata*training_pc)
    test       = int(ndata*test_pc)
    validation = int(ndata-training-test)

    # Indices for training
    idx_train = idx[0:training]
    # Indices for testing
    idx_test  = idx[training:training+test]
    # Indices for validation
    idx_val   = idx[training+test:]

    # Training set
    training_data = {
     "obs_traj":      data["obs_traj"][idx_train],
     "obs_traj_rel":  data["obs_traj_rel"][idx_train],
     "pred_traj":     data["pred_traj"][idx_train],
     "pred_traj_rel": data["pred_traj_rel"][idx_train]
    }

    # Test set
    test_data = {
     "obs_traj":     data["obs_traj"][idx_test],
     "obs_traj_rel": data["obs_traj_rel"][idx_test],
     "pred_traj":    data["pred_traj"][idx_test],
     "pred_traj_rel":data["pred_traj_rel"][idx_test]
    }

    # Validation set
    validation_data ={
     "obs_traj":     data["obs_traj"][idx_val],
     "obs_traj_rel": data["obs_traj_rel"][idx_val],
     "pred_traj":    data["pred_traj"][idx_val],
     "pred_traj_rel":data["pred_traj_rel"][idx_val]
    }


    print("[INF] Training data: "+ str(len(training_data[list(training_data.keys())[0]])))
    print("[INF] Test data: "+ str(len(test_data[list(test_data.keys())[0]])))
    print("[INF] Validation data: "+ str(len(validation_data[list(validation_data.keys())[0]])))

    # Plot ramdomly a subset of the training data (spatial data only)
    nSamples = min(30,training)
    samples  = random.sample(range(1,training), nSamples)
    plt.subplots(1,1,figsize=(10,10))
    plt.subplot(1,1,1)
    plt.axis('equal')
    # Plot some of the training data
    for (o,p) in zip(training_data["obs_traj"][samples],training_data["pred_traj"][samples]):
        # Observations
        plt.plot(o[:,0],o[:,1],color='red')
        plt.plot([o[-1,0],p[0,0]],[o[-1,1],p[0,1]],color='blue')
        # Prediction targets
        plt.plot(p[:,0],p[:,1],color='blue')
        plt.title("Samples of trajectories from the dataset")
        plt.show()


if __name__ == '__main__':
    main()
