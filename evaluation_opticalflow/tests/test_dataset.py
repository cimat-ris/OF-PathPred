import sys,os
sys.path.append('../lib')
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
print('Tensorflow version: ',tf.__version__)
tf.test.gpu_device_name()
import random
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

# Important imports
from process_file import process_file, predictor_parameters
from process_file import process_file_modif_varios
import batches_data

# Dataset to be tested
dataset_paths  = "../../data1/"
#dataset_name = 'eth-hotel'
#dataset_name = 'eth-univ'
#dataset_name = 'ucy-zara01'
dataset_name = 'ucy-zara02'
# File of trajectories coordinates. Coordinates are in world frame
data_path = dataset_paths+dataset_name

# Load the default parameters
parameters = predictor_parameters()
# Process data specified by the path to get the trajectories with
data = process_file(data_path, parameters, ',')

# Seed
random.seed(datetime.now())
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
     "pred_traj_rel": data["pred_traj_rel"][idx_train],
     "key_idx":       data["key_idx"][[idx_train]],
}

# Test set
test_data = {
     "obs_traj":     data["obs_traj"][idx_test],
     "obs_traj_rel": data["obs_traj_rel"][idx_test],
     "pred_traj":    data["pred_traj"][idx_test],
     "pred_traj_rel":data["pred_traj_rel"][idx_test],
     "key_idx":      data["key_idx"][[idx_test]],
}

# Validation set
validation_data ={
     "obs_traj":     data["obs_traj"][idx_val],
     "obs_traj_rel": data["obs_traj_rel"][idx_val],
     "pred_traj":    data["pred_traj"][idx_val],
     "pred_traj_rel":data["pred_traj_rel"][idx_val],
     "key_idx":      data["key_idx"][[idx_val]],
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
plt.title("Samples of trajectories from "+dataset_name)
plt.show()
