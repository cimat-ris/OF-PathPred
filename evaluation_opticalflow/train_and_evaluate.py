# Imports
import sys,os
sys.path.append('./lib')
import math,numpy as np
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
print('Tensorflow version: ',tf.__version__)
tf.test.gpu_device_name()
import random
from datetime import datetime
random.seed(datetime.now())

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

# Important imports
from process_file import process_file
import batches_data
from model import Model, Model_Parameters
from training_and_testing import Trainer,Tester,Experiment_Parameters
import matplotlib.pyplot as plt


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
plt.show()

import pickle

# Training dataset
pickle_out = open('training_data_'+dataset_name+'.pickle',"wb")
pickle.dump(training_data, pickle_out, protocol=2)
pickle_out.close()

# Test dataset
pickle_out = open('test_data_'+dataset_name+'.pickle',"wb")
pickle.dump(test_data, pickle_out, protocol=2)
pickle_out.close()

# Validation dataset
pickle_out = open('validation_data_'+dataset_name+'.pickle',"wb")
pickle.dump(validation_data, pickle_out, protocol=2)
pickle_out.close()

from tqdm import tqdm
tf.reset_default_graph()

model_parameters = Model_Parameters(train_num_examples=len(training_data['obs_traj']),add_kp = experiment_parameters.add_kp, add_social=experiment_parameters.add_social)
model            = Model(model_parameters)
train_data       = batches_data.Dataset(training_data,model_parameters)
val_data         = batches_data.Dataset(validation_data,model_parameters)

saver     = tf.train.Saver(max_to_keep = 2)
bestsaver = tf.train.Saver(max_to_keep = 2)

trainer   = Trainer(model,model_parameters)
tester    = Tester(model, model_parameters)

# Global variables are initialized
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

increment_global_step_op = tf.assign(model.global_step, model.global_step+1)

val_perf  = []
loss      = -1
best      = {'ade':999999, 'fde':0, 'step':-1}
is_start  = True
num_steps = int(math.ceil(train_data.num_examples/float(model_parameters.batch_size)))
loss_list = []

print("[INF] Training")
# Epochs
for i in range(model_parameters.num_epochs):
    # Cycle over batches
    for idx, batch in tqdm(train_data.get_batches(model_parameters.batch_size,num_steps = num_steps),total=num_steps):
        # Increment global step
        sess.run(increment_global_step_op)
        global_step = sess.run(model.global_step)

        # Evaluation on validation data
        if((global_step%model_parameters.validate==0) or (model_parameters.load_best and is_start)):
            checkpoint_path_model = os.path.join('models/'+dataset_name, 'model.ckpt')
            saver.save(sess,checkpoint_path_model , global_step = global_step)
            # Evaluation on th validation set
            results = tester.evaluate(val_data,sess)
            if results["ade"]< best['ade']:
                best['ade'] = results["ade"]
                best['fde'] = results["fde"]
                best["step"]= global_step
                # Save the best model
                checkpoint_path_model_best = os.path.join('models/'+dataset_name, 'model_best.ckpt')
                bestsaver.save(sess,checkpoint_path_model_best,global_step = 0)
                finalperf = results
                val_perf.append((loss, results))
            is_start = False
        loss, train_op = trainer.step(sess, batch)
        loss_list.append(loss)

if((global_step % model_parameters.validate)!=0):
  checkpoint_path_model = os.path.join('models/'+dataset_name, 'model.ckpt')
  saver.save(sess,checkpoint_path_model , global_step = global_step)

print("best eval on val %s: %s at %s step y fde es %s " % ('ade', best['ade'], best["step"],best["fde"]))

from matplotlib import pyplot as plt
plt.figure(figsize=(8,8))
plt.subplot(1,1,1)
plt.plot(loss_list)
plt.show()
# Keep the last model
checkpoint_path_model = os.path.join('models/'+dataset_name, 'lastmodel.ckpt')
saver.save(sess,checkpoint_path_model , global_step = 0)

# TESTING
print("[INF] Testing")
# Select one random batch
test_batches_data = batches_data.Dataset(test_data, model_parameters)
batchId  = random.randint(1,test_batches_data.get_num_batches())

# Load the last model that was saved
path_model = 'models/'+dataset_name+'/lastmodel.ckpt-0'
saver.restore(sess=sess, save_path=path_model)
results           = tester.evaluate(test_batches_data,sess)
# Qualitative evaluation: test on batch batchId
traj_obs_set,traj_gt_set,traj_pred_set = tester.apply_on_batch(test_batches_data,batchId,sess)
plt.subplots(1,1,figsize=(10,10))
plt.subplot(1,1,1)
plt.axis('equal')
# Plot some of the testing data and the predicted ones
for (gt,obs,pred) in zip(traj_gt_set,traj_obs_set,traj_pred_set):
    plt.plot(obs[:,0],obs[:,1],color='red')
    # Ground truth trajectory
    plt.plot([obs[-1,0],gt[0,0]],[obs[-1,1],gt[0,1]],color='blue')
    plt.plot(gt[:,0],gt[:,1],color='blue')
    # Predicted trajectory
    plt.plot([obs[-1,0],pred[0,0]],[obs[-1,1],pred[0,1]],color='green')
    plt.plot(pred[:,0],pred[:,1],color='green')
plt.show()

# Best model
path_model = 'models/'+dataset_name+'/model_best.ckpt-0'
saver.restore(sess=sess, save_path=path_model)
results           = tester.evaluate(test_batches_data,sess)
# Qualitative evaluation: test on batch batchId
traj_obs_set,traj_gt_set,traj_pred_set = tester.apply_on_batch(test_batches_data,batchId,sess)
plt.subplots(1,1,figsize=(10,10))
plt.subplot(1,1,1)
plt.axis('equal')
# Plot some of the testing data and the predicted ones
for (gt,obs,pred) in zip(traj_gt_set,traj_obs_set,traj_pred_set):
    plt.plot(obs[:,0],obs[:,1],color='red')
    plt.plot([obs[-1,0],gt[0,0]],[obs[-1,1],gt[0,1]],color='blue')
    plt.plot(gt[:,0],gt[:,1],color='blue')
    plt.plot([obs[-1,0],pred[0,0]],[obs[-1,1],pred[0,1]],color='green')
    plt.plot(pred[:,0],pred[:,1],color='green')
plt.show()
