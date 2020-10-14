# Imports
import sys,os
sys.path.append('./lib')
import math,numpy as np
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
print('Tensorflow version: ',tf.__version__)
tf.test.gpu_device_name()
# Important imports
from process_file import process_file
import batches_data
from training_and_testing import Trainer,Tester,Experiment_Parameters
import matplotlib.pyplot as plt
from model import TrajectoryEncoderDecoder, Model_Parameters
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models



# Load the default parameters
experiment_parameters = Experiment_Parameters(add_social=True,add_kp=False,obstacles=True)

# Dataset to be tested
dataset_paths  = "../data1/"
#dataset_name = 'eth-hotel'
#dataset_name = 'eth-univ'
#dataset_name = 'ucy-zara01'
dataset_name = 'ucy-zara02'
recompute_opticalflow = False
# File of trajectories coordinates. Coordinates are in world frame
data_path = dataset_paths+dataset_name

# Process data specified by the path to get the trajectories with
print('[INF] Extracting data from the ddatasets')
data = process_file(data_path, experiment_parameters, ',')

# Should be nSamples x sequenceLength x nPersonsMax x PersonDescriptionSize
if experiment_parameters.add_social:
    print(data['obs_neighbors'].shape)

# Muestreamos aleatoriamente para separar datos de entrenamiento, validacion y prueba
training_pc  = 0.7
test_pc      = 0.2

# Count how many data we have (sub-sequences of length 8, in pred_traj)
ndata      = len(data[list(data.keys())[2]])
idx        = np.random.permutation(ndata)
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
     "obs_flow":      data["obs_flow"][idx_train]
}

# Test set
test_data = {
     "obs_traj":     data["obs_traj"][idx_test],
     "obs_traj_rel": data["obs_traj_rel"][idx_test],
     "pred_traj":    data["pred_traj"][idx_test],
     "pred_traj_rel":data["pred_traj_rel"][idx_test],
     "key_idx":      data["key_idx"][[idx_test]],
     "obs_flow":     data["obs_flow"][idx_test]
}

# Validation set
validation_data ={
     "obs_traj":     data["obs_traj"][idx_val],
     "obs_traj_rel": data["obs_traj_rel"][idx_val],
     "pred_traj":    data["pred_traj"][idx_val],
     "pred_traj_rel":data["pred_traj_rel"][idx_val],
     "key_idx":      data["key_idx"][[idx_val]],
     "obs_flow":     data["obs_flow"][idx_val]
}


print("[INF] Training data: "+ str(len(training_data[list(training_data.keys())[0]])))
print("[INF] Test data: "+ str(len(test_data[list(test_data.keys())[0]])))
print("[INF] Validation data: "+ str(len(validation_data[list(validation_data.keys())[0]])))

# Model
model_parameters = Model_Parameters(train_num_examples=1,add_kp=False,add_social=True)
# x is NxT_obsx2 (simulation of a batch of trajectories)
x = tf.ones((3,10,2))
# y is NxT_predx2 (simulation of a batch of trajectories)
y = tf.cumsum(tf.ones((3,12,2)),axis=1)
# x is NxT_obsx20 (simulation of a batch of social features)
s = tf.ones((3,10,20))
tj_enc_dec = TrajectoryEncoderDecoder(model_parameters)
tj_enc_dec.build(input_shape=[(None,10,model_parameters.P),(None,12,model_parameters.P)])
tj_enc_dec.compile(optimizer='Adam', loss="mse", metrics=["mae"])
tj_enc_dec.summary()
xp     = tj_enc_dec([x,y])


train_data       = batches_data.Dataset(training_data,model_parameters)
val_data         = batches_data.Dataset(validation_data,model_parameters)

# Run training
#model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
#          batch_size=batch_size,
#          epochs=epochs,
#          validation_split=0.2)
