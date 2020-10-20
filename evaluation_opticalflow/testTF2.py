# Imports
import sys,os
sys.path.append('./lib')
import math,numpy as np
import warnings
from tqdm import tqdm
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


def get_batch(batch_data, config, train=False):
    """Given a batch of data, determine the input and ground truth."""
    N      = len(batch_data['obs_traj_rel'])
    P      = config.P
    OF     = config.flow_size
    T_in   = config.obs_len
    T_pred = config.pred_len

    returned_inputs = []
    traj_obs_gt  = np.zeros([N, T_in, P], dtype='float32')
    traj_pred_gt = np.zeros([N, T_pred, P], dtype='float32')
    # --- xy input
    for i, (obs_data, pred_data) in enumerate(zip(batch_data['obs_traj_rel'],
                                                  batch_data['pred_traj_rel'])):
        for j, xy in enumerate(obs_data):
            traj_obs_gt[i, j, :] = xy
        for j, xy in enumerate(pred_data):
            traj_pred_gt[i, j, :]   = xy
    returned_inputs.append(traj_obs_gt)
    # ------------------------------------------------------
    # Social component (through optical flow)
    if config.add_social:
        obs_flow = np.zeros((N, T_in, OF),dtype ='float32')
        # each batch
        for i, flow_seq in enumerate(batch_data['obs_flow']):
            for j , flow_step in enumerate(flow_seq):
                obs_flow[i,j,:] = flow_step
        returned_inputs.append(obs_flow)
    # -----------------------------------------------------------
    # Person pose input
    if config.add_kp:
        obs_kp = np.zeros((N, T_in, KP, 2), dtype='float32')
        # each bacth
        for i, obs_kp_rel in enumerate(batch_data['obs_kp_rel']):
            for j, obs_kp_step in enumerate(obs_kp_rel):
                obs_kp[i, j, :, :] = obs_kp_step
    return returned_inputs,traj_pred_gt

# Load the default parameters
experiment_parameters = Experiment_Parameters(add_social=False,add_kp=False,obstacles=True)

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
print('[INF] Extracting data from thedatasets')
data = process_file(data_path, experiment_parameters, ',')

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
     "key_idx":       data["key_idx"][idx_train]
}
if experiment_parameters.add_social:
    training_data["obs_flow"]=data["obs_flow"][idx_train]

# Test set
test_data = {
     "obs_traj":     data["obs_traj"][idx_test],
     "obs_traj_rel": data["obs_traj_rel"][idx_test],
     "pred_traj":    data["pred_traj"][idx_test],
     "pred_traj_rel":data["pred_traj_rel"][idx_test],
     "key_idx":      data["key_idx"][idx_test]
}
if experiment_parameters.add_social:
    test_data["obs_flow"]=data["obs_flow"][idx_test]

# Validation set
validation_data ={
     "obs_traj":     data["obs_traj"][idx_val],
     "obs_traj_rel": data["obs_traj_rel"][idx_val],
     "pred_traj":    data["pred_traj"][idx_val],
     "pred_traj_rel":data["pred_traj_rel"][idx_val],
     "key_idx":      data["key_idx"][idx_val]
}
if experiment_parameters.add_social:
    validation_data["obs_flow"]=data["obs_flow"][idx_val]

print("[INF] Training data: "+ str(len(training_data[list(training_data.keys())[0]])))
print("[INF] Test data: "+ str(len(test_data[list(test_data.keys())[0]])))
print("[INF] Validation data: "+ str(len(validation_data[list(validation_data.keys())[0]])))

# Model
model_parameters = Model_Parameters(train_num_examples=1,add_kp=False,add_social=False)
# x is NxT_obsx2 (simulation of a batch of trajectories)
x = tf.ones((3,model_parameters.obs_len,model_parameters.P))
# y is NxT_predx2 (simulation of a batch of trajectories)
y = tf.cumsum(tf.ones((3,model_parameters.pred_len,model_parameters.P)),axis=1)
# x is NxT_obsx20 (simulation of a batch of social features)
s = tf.ones((3,model_parameters.obs_len,model_parameters.flow_size))

obs_shape  = (model_parameters.obs_len,model_parameters.P)
gt_shape   = (model_parameters.pred_len,model_parameters.P)
soc_shape  = (model_parameters.obs_len,model_parameters.flow_size)
tj_enc_dec = TrajectoryEncoderDecoder(model_parameters)

train_data       = batches_data.Dataset(training_data,model_parameters)
val_data         = batches_data.Dataset(validation_data,model_parameters)

num_batches_per_epoch = train_data.get_num_batches()
print("[INF] Training")
train_loss_results = []
# Epochs
for epoch in range(model_parameters.num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    # Cycle over batches
    num_batches_per_epoch = train_data.get_num_batches()
    for idx, batch in tqdm(train_data.get_batches(model_parameters.batch_size, num_steps = num_batches_per_epoch, shuffle=True), total = num_batches_per_epoch, ascii = True):
        # Format the data
        batch_inputs, batch_targets = get_batch(batch, model_parameters, train=True)
        # Run the forward pass of the layer.
        # Compute the loss value for this minibatch.
        loss_value = tj_enc_dec.train_step(batch_inputs, batch_targets)  # Logits for this minibatch
        # Log every 200 batches.
        #if step % 200 == 0:
    #        print("[INF] Training loss (for one batch) at step %d: %.4f"
    #            % (step, float(loss_value))
    #            )
    #        print("[INF] Seen so far: %s samples" % ((step + 1) * 64))


    # End epoch
    train_loss_results.append(epoch_loss_avg.result())
    if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}".format(epoch,epoch_loss_avg.result()))
