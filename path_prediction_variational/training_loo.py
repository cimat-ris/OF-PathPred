# Imports
import sys,os
''' TF_CPP_MIN_LOG_LEVEL
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append('./lib')
import math,numpy as np
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
print('[INF] Tensorflow version: ',tf.__version__)
tf.test.gpu_device_name()
# Important imports
import batches_data
#import matplotlib.pyplot as plt
from model import TrajectoryEncoderDecoder, Model_Parameters
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from datasets_utils import setup_loo_experiment, get_testing_batch
from plot_utils import plot_training_data,plot_training_results
from testing_utils import evaluation_minadefde,evaluation_qualitative,evaluation_attention,plot_comparisons_minadefde
from training_utils import training_loop
from training_utils import Experiment_Parameters


physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices)>0:
    print('[INF] Using GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("[INF] Using CPU")

# Load the default parameters
experiment_parameters = Experiment_Parameters(add_social=False,add_kp=False,obstacles=False)

dataset_dir   = "../datasets/"
dataset_names = ['eth-hotel','eth-univ','ucy-zara01','ucy-zara02','ucy-univ']

# Load the dataset and perform the split
seed = 0
idTest = 2
training_data,validation_data,test_data,test_homography = setup_loo_experiment('ETH_UCY',dataset_dir,dataset_names,idTest,experiment_parameters,use_pickled_data=False, seed=seed)

# Plot ramdomly a subset of the training data (spatial data only)
show_training_samples = False
if show_training_samples:
    plot_training_data(training_data,experiment_parameters)

#############################################################
# Model parameters
model_parameters = Model_Parameters(add_attention=True,add_kp=experiment_parameters.add_kp,add_social=experiment_parameters.add_social,output_representation=experiment_parameters.output_representation)

model_parameters.num_epochs     = 36
model_parameters.output_var_dirs= 0
model_parameters.is_mc_dropout  = False
model_parameters.initial_lr     = 0.5 #0.03
model_parameters.dropout_rate = 0

# Get the necessary data
train_data = tf.data.Dataset.from_tensor_slices(training_data)
val_data   = tf.data.Dataset.from_tensor_slices(validation_data)
test_data  = tf.data.Dataset.from_tensor_slices(test_data)

print(training_data.keys())
print(training_data['obs_traj'].shape)
print(training_data['obs_traj_rel'].shape)
print(training_data['obs_traj_theta'].shape)
print(training_data['pred_traj'].shape)
print(training_data['pred_traj_rel'].shape)
print(training_data['frames_ids'].shape)


# Form batches
batched_train_data = train_data.batch(model_parameters.batch_size)
batched_val_data   = val_data.batch(model_parameters.batch_size)
batched_test_data  = test_data.batch(model_parameters.batch_size)

#------
seed = 0
tf.random.set_seed(seed)
#------

# Model
#tf.random.set_seed(2)
tj_enc_dec = TrajectoryEncoderDecoder(model_parameters)

# Checkpoints
checkpoint_dir   = './training_checkpoints'
checkpoint_prefix= os.path.join(checkpoint_dir, "ckpt")
checkpoint       = tf.train.Checkpoint(optimizer=tj_enc_dec.optimizer,
                                        encoder=tj_enc_dec.enc,
                                        decoder=tj_enc_dec.dec,
                                        enctodec=tj_enc_dec.enctodec,
                                        obs_classif=tj_enc_dec.obs_classif)

# Training
perform_training = True
plot_training    = True
if perform_training==True:
    print("[INF] Training the model")
    train_loss_results,val_loss_results,val_metrics_results,__ = training_loop(tj_enc_dec,batched_train_data,batched_val_data,model_parameters,checkpoint,checkpoint_prefix)
    if plot_training==True:
        plot_training_results(train_loss_results,val_loss_results,val_metrics_results)

# Testing
# Restoring the latest checkpoint in checkpoint_dir
print("[INF] Restoring last model")
status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
import pickle
pickle_in = open('training_cSGMCMC_weights.pickle',"rb")
weights = pickle.load(pickle_in)
print("len(weights): ",len(weights))
print("print last layer of decoder:")
for www in weights:
    print(www[0], www[1][2][6])


#######################################################
# Quantitative testing: ADE/FDE/NLL
from testing_utils import evaluation_minadefde_nll

print("[INF] Quantitative testing")
results = evaluation_minadefde_nll(tj_enc_dec,batched_test_data,model_parameters, weights)
print("Results:")
print(results)


######################################################
from plot_utils import plot_gt_preds, plot_background, plot_neighbors, plot_attention
#from testing_utils import predict_from_batch
from batches_data import get_batch
from traj_utils import relative_to_abs

from testing_utils import compute_kde_nll2

# Qualitative testing
num_print_test = 30
model_parameters.output_var_dirs= 0
qualitative = True
if qualitative==True:
    print("[INF] Qualitative testing")
    for it in range(num_print_test):
        batch, test_bckgd = get_testing_batch(test_data,dataset_dir+dataset_names[idTest])
        print("batch: ", batch['obs_traj'].shape)
        
        # Plot ground truth and predictions
        plt.subplots(1,1,figsize=(10,10))
        ax = plt.subplot(1,1,1)
        if test_bckgd is not None:
            plot_background(ax,test_bckgd)

        #-----------------------------------------------------------------------
        batch_inputs, batch_targets = get_batch(batch, model_parameters)

        # Iterate the diference weights for the model
        cont = 0
        for w_ind, weight in weights:
          # Perform prediction
          pred_traj, _ = tj_enc_dec.predict_cSGMCMC(batch_inputs,batch_targets.shape[1],weight)

          traj_obs      = []
          traj_gt       = []
          traj_pred     = []
          neighbors     = []
          # Cycle over the trajectories of the bach
          for i, (obs_traj_gt, pred_traj_gt, neighbors_gt) in enumerate(zip(batch["obs_traj"], batch["pred_traj"], batch["obs_neighbors"])):
              this_pred_out_abs_set = []
              # Conserve the x,y coordinates
              #print("pred_traj[i,0].shape[0]: ", pred_traj[i,0].shape[0])
              #print("model_parameters.pred_len: ", model_parameters.pred_len)
              #print("pred_traj.shape: ", pred_traj.shape)

              if (pred_traj[i,0].shape[0] == model_parameters.pred_len):
                  this_pred_out     = pred_traj[i,0,:, :2]
                  # Convert it to absolute (starting from the last observed position)
                  this_pred_out_abs = relative_to_abs(this_pred_out, obs_traj_gt[-1])
                  this_pred_out_abs_set.append(this_pred_out_abs)
               #   print("this_pred_out_abs: ", this_pred_out_abs.shape)
              this_pred_out_abs_set = tf.stack(this_pred_out_abs_set,axis=0)
              # TODO: tensors instead of lists?
              # Keep all the trajectories
              traj_obs.append(obs_traj_gt)
              traj_gt.append(pred_traj_gt)
              traj_pred.append(this_pred_out_abs_set)
              neighbors.append(neighbors_gt)

          #-----------------------------------------------------------------------
          # Plot ground truth and predictions
          if cont == 0:
            pred_out = np.array(traj_pred)
          else:
            pred_out = np.concatenate([pred_out, np.array(traj_pred)],axis=1)
          cont += 1

          plot_neighbors(ax,neighbors,test_homography,flip=False)
          plot_gt_preds(ax,traj_gt,traj_obs,traj_pred,test_homography,flip=False,display_mode=None,n_peds_max=1,mode=w_ind)
        plt.savefig('images/tmp_'+str(it)+'.pdf')
        plt.show()

        kde_nll, timestep_kde_nll = compute_kde_nll2(pred_out, np.array(traj_gt))
        print("kde_nll: ", kde_nll)
        print("timestep_kde_nll: ", timestep_kde_nll)


# Quantitative testing: ADE/FDE
#quantitative = False
#if quantitative==True:
#    print("[INF] Quantitative testing")
#    results = evaluation_minadefde(tj_enc_dec,batched_test_data,model_parameters)
#    plot_comparisons_minadefde(results,dataset_names[idTest])
#    print(results)

# Qualitative testing
#qualitative = True
#if qualitative==True:
#    print("[INF] Qualitative testing")
#    for i in range(5):
#        batch, test_bckgd = get_testing_batch(test_data,dataset_dir+dataset_names[idTest])
        #evaluation_qualitative(tj_enc_dec,batch,model_parameters,background=test_bckgd,homography=test_homography, flip=False,n_peds_max=1,display_mode=None)
#        evaluation_attention(tj_enc_dec,batch,model_parameters,background=test_bckgd,homography=test_homography, flip=False,display_mode=None)
