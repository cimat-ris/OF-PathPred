# Imports
import sys,os
sys.path.append('./lib')
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
print('Tensorflow version: ',tf.__version__)
tf.test.gpu_device_name()
import random

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

# Important imports
from process_file import process_file_modif
from process_file import process_file_modif_varios
import batches_data
import model
import entrenamientoevaluacion
from interaction_optical_flow import OpticalFlowSimulator

# Dataset to be tested
dataset_paths  = "../data1/"
#dataset_name = 'eth-hotel'
#dataset_name = 'eth-univ'
#dataset_name = 'ucy-zara01'
dataset_name = 'ucy-zara02'
recompute_opticalflow = False
# To test obstacle-related functions
from obstacles import image_to_world_xy,generate_obstacle_polygons,load_world_obstacle_polygons
import matplotlib.pyplot as plt

# Determine a list of obstacles for this dataset, from the semantic map and save the results
generate_obstacle_polygons(dataset_paths,dataset_name)
# Load the saved obstacles
obstacles_world = load_world_obstacle_polygons(dataset_paths,dataset_name)

# File of trajectories coordinates. Coordinates are in world frame
data_path = '../data1/'+dataset_name

# Parameters
class parameters:
    def __init__(self):
        # Maximum number of persons in a frame
        self.person_max = 42 # 8   # Univ: 42  Hotel: 28
        # Observation length (trajlet size)
        self.obs_len    = 8
        # Prediction length
        self.pred_len   = 12
        # Flag to consider social interactions
        self.add_social = True
        # Number of key points
        self.kp_num     = 18
        # Key point flag
        self.add_kp     = False
        # Obstacles flag
        self.obstacles  = True

# Load the default parameters
arguments = parameters()

# Process data to get the trajectories
data = process_file_modif(data_path, arguments, ',')

# Should be nSamples x sequenceLength x nPersonsMax x PersonDescriptionSize
print(data['obs_person'].shape)

# Optical flow
optical_flow_file = 'of_'+dataset_name+'.npy'
OFSimulator = OpticalFlowSimulator()
if recompute_opticalflow==True:
    # Compute the optical flow generated by the neighbors
    optical_flow, __, __ = OFSimulator.compute_opticalflow_batch(data['obs_person'],data['key_idx'],data['obs_traj'],arguments.obs_len,obstacles_world)
    # Save it on file
    np.save(optical_flow_file,optical_flow)

# Load the optical flow
optical_flow = np.load(optical_flow_file)
data.update({"obs_flow": optical_flow})

# Seed
random.seed(0)

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


print("training data: "+ str(len(training_data[list(training_data.keys())[0]])))
print("test data: "+ str(len(test_data[list(test_data.keys())[0]])))
print("validation data: "+ str(len(validation_data[list(validation_data.keys())[0]])))

import matplotlib.pyplot as plt

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


import pickle

# Training dataset
pickle_out = open('training_data_'+dataset_name+'.pickle',"wb")
pickle.dump(training, pickle_out, protocol=2)
pickle_out.close()

# Test dataset
pickle_out = open('test_data_'+dataset_name+'.pickle',"wb")
pickle.dump(test, pickle_out, protocol=2)
pickle_out.close()

# Validation dataset
pickle_out = open('validation_data_'+dataset_name+'.pickle',"wb")
pickle.dump(validation, pickle_out, protocol=2)
pickle_out.close()

class model_parameters:
    def __init__(self, train_num_examples, add_kp = False, add_social = False):
        # -----------------
        # Observation/prediction lengths
        self.obs_len  = 8
        self.pred_len = 12

        self.add_kp             = add_kp
        self.train_num_examples = train_num_examples
        self.add_social         = add_social
        # Key points
        self.kp_num = 18
        self.kp_size = 18
        #self.maxNumPed = 8
        #self.grid_size = 4
        #self.neighborhood_size = 32
        #self.dimensions = [768,576]
        #self.limites=[-15.88,11.56,-19.09,5.99]
        #self.bound=[0.7,0.5]
        # ------------------
        self.num_epochs = 100
        self.batch_size = 20 # batch size
        self.validate   = 300
        self.P          = 2 # Dimension
        self.enc_hidden_size = 64 # el nombre lo dice
        self.dec_hidden_size = 64
        self.emb_size        = 64
        self.keep_prob      = 0.7 # dropout

        self.min_ped = 1
        self.seq_len = self.obs_len + self.pred_len
        self.reverse_xy = False

        self.activation_func = tf.nn.tanh
        self.activation_func1 = tf.nn.relu
        self.is_train = True
        self.is_test = False
        self.multi_decoder = False
        self.modelname = 'gphuctl'

        self.init_lr = 0.002 # 0.01
        self.learning_rate_decay = 0.85
        self.num_epoch_per_decay = 2.0
        self.optimizer = 'adam'
        self.emb_lr = 1.0
        #self.clip_gradient_norm = 10.0
        #Para cuando entreno y quiero guardar el mejor modelo
        self.load_best = True


import os
from tqdm import tqdm
import tensorflow as tf
import math
import model
tf.reset_default_graph()

arguments = model_parameters(train_num_examples=len(training_data['obs_traj']),add_kp = False, add_social = True,)
model     = model.Model(arguments)

train_data = batches_data.Dataset(training_data,arguments)
val_data   = batches_data.Dataset(validation_data,arguments)

saver     = tf.train.Saver(max_to_keep = 2)
bestsaver = tf.train.Saver(max_to_keep = 2)


trainer = entrenamientoevaluacion.Trainer(model,arguments)
tester  = entrenamientoevaluacion.Tester(model, arguments)

# Global variables are initialized
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

increment_global_step_op = tf.assign(model.global_step, model.global_step+1)

val_perf  = []
loss      = -1
best      = {'ade':999999, 'fde':0, 'step':-1}
is_start  = True
num_steps = int(math.ceil(train_data.num_examples/float(arguments.batch_size)))
loss_list = []

# Epochs
for i in range(arguments.num_epochs):
    # Cycle over batches
    for idx, batch in tqdm(train_data.get_batches(arguments.batch_size,num_steps = num_steps),total=num_steps):

        sess.run(increment_global_step_op)
        global_step = sess.run(model.global_step)

        # Evaluation on validation data
        if((global_step%arguments.validate==0) or (arguments.load_best and is_start)):
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

if((global_step % arguments.validate)!=0):
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
# Load the last model that was saved
path_model = 'models/'+dataset_name+'/lastmodel.ckpt-0'
saver.restore(sess=sess, save_path=path_model)
test_batches_data = batches_data.Dataset(test_data, arguments)
results           = tester.evaluate(test_batches_data,sess)
print(results)

nBatches = int(math.ceil(test_batches_data.num_examples / float(arguments.batch_size)))
batchId  = random.sample(range(1,nBatches), 1)

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

# Best model
path_model = 'models/'+dataset_name+'/model_best.ckpt-0'
saver.restore(sess=sess, save_path=path_model)

test_batches_data = batches_data.Dataset(test_data, arguments)
results  = tester.evaluate(test_batches_data,sess)
print(results)

# Apply the best model
nBatches = int(math.ceil(test_batches_data.num_examples / float(arguments.batch_size)))
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
