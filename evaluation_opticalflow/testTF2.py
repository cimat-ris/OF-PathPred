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
dataset_dir               = "../data1/"
testing_data_paths        = [dataset_dir+'eth-univ']
training_data_paths       = [dataset_dir+'eth-hotel',dataset_dir+'ucy-zara01',dataset_dir+'ucy-zara02',dataset_dir+'ucy-univ']

# Process data specified by the path to get the trajectories with
print('[INF] Extracting data from the datasets')
test_data     = process_file(testing_data_paths, experiment_parameters)
train_data = process_file(training_data_paths, experiment_parameters)

# Count how many data we have (sub-sequences of length 8, in pred_traj)
n_test_data  = len(test_data[list(test_data.keys())[2]])
n_train_data = len(train_data[list(train_data.keys())[2]])
idx        = np.random.permutation(n_train_data)
validation_pc = 0.1
validation    = int(n_train_data*validation_pc)
training      = int(n_train_data-validation)

# Indices for training
idx_train = idx[0:training]
# Indices for validation
idx_val   = idx[training:]
# Training set
training_data = {
     "obs_traj":      train_data["obs_traj"][idx_train],
     "obs_traj_rel":  train_data["obs_traj_rel"][idx_train],
     "pred_traj":     train_data["pred_traj"][idx_train],
     "pred_traj_rel": train_data["pred_traj_rel"][idx_train],
}
if experiment_parameters.add_social:
    training_data["obs_flow"]=train_data["obs_flow"][idx_train]
# Test set
testing_data = {
     "obs_traj":     test_data["obs_traj"][:],
     "obs_traj_rel": test_data["obs_traj_rel"][:],
     "pred_traj":    test_data["pred_traj"][:],
     "pred_traj_rel":test_data["pred_traj_rel"][:],
}
if experiment_parameters.add_social:
    testing_data["obs_flow"]=test_data["obs_flow"][:]

# Validation set
validation_data ={
     "obs_traj":     train_data["obs_traj"][idx_val],
     "obs_traj_rel": train_data["obs_traj_rel"][idx_val],
     "pred_traj":    train_data["pred_traj"][idx_val],
     "pred_traj_rel":train_data["pred_traj_rel"][idx_val],
}
if experiment_parameters.add_social:
    validation_data["obs_flow"]=train_data["obs_flow"][idx_val]
print("[INF] Training data: "+ str(len(training_data[list(training_data.keys())[0]])))
print("[INF] Test data: "+ str(len(test_data[list(test_data.keys())[0]])))
print("[INF] Validation data: "+ str(len(validation_data[list(validation_data.keys())[0]])))

# Model parameters
model_parameters = Model_Parameters(add_kp=False,add_social=True)
model_parameters.num_epochs = 100

# Get the necessary data
train_data       = batches_data.Dataset(training_data,model_parameters)
val_data         = batches_data.Dataset(validation_data,model_parameters)
test_data        = batches_data.Dataset(test_data, model_parameters)

# Model
tj_enc_dec = TrajectoryEncoderDecoder(model_parameters)


# Checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=tj_enc_dec.optimizer,
                                    encoder=tj_enc_dec.enc,
                                    decoder=tj_enc_dec.dec)

# Training
print("[INF] Training")
perform_training = True
if perform_training==True:
    train_loss_results,val_loss_results,val_metrics_results,__ = tj_enc_dec.training_loop(train_data,val_data,model_parameters,checkpoint,checkpoint_prefix)
    # Plot training results
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(train_loss_results,'b',label='Training')
    ax.plot(val_loss_results,'r',label='Validation')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title('Training and validation losses')
    ax.legend()
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(val_metrics_results["ade"],'b',label='ADE in validation')
    ax.plot(val_metrics_results["fde"],'r',label='FDE in validation')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error (m)")
    ax.set_title('Metrics at validation')
    ax.legend()
    plt.show()


# Testing
# Restoring the latest checkpoint in checkpoint_dir
print("[INF] Restoring last model")
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
# Quantitative testing: ADE/FDE
print("[INF] Quantitative testing")
results = tj_enc_dec.quantitative_evaluation(test_data,model_parameters)
print(results)
# Qualitative testing
print("[INF] Qualitative testing")
tj_enc_dec.qualitative_evaluation(test_data,model_parameters,10)
