# Imports
import sys,os
''' TF_CPP_MIN_LOG_LEVEL
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''
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
import matplotlib.pyplot as plt
from model import TrajectoryEncoderDecoder, Model_Parameters
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from datasets_utils import setup_loo_experiment, get_testing_batch
from plot_utils import plot_training_data,plot_training_results
from testing_utils import evaluation_minadefde,evaluation_qualitative,plot_comparisons_minadefde
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
idTest = 2
training_data,validation_data,test_data,test_homography = setup_loo_experiment('ETH_UCY',dataset_dir,dataset_names,idTest,experiment_parameters,use_pickled_data=False)

# Plot ramdomly a subset of the training data (spatial data only)
show_training_samples = False
if show_training_samples:
    plot_training_data(training_data,experiment_parameters)

#############################################################
# Model parameters
model_parameters = Model_Parameters(add_attention=True,add_kp=experiment_parameters.add_kp,add_social=experiment_parameters.add_social,output_representation=experiment_parameters.output_representation)
if experiment_parameters.output_representation == 'vw':
    model_parameters.num_epochs = 100
    model_parameters.initial_lr = 0.1
model_parameters.num_epochs     = 1
model_parameters.output_var_dirs= 4
model_parameters.is_mc_dropout  = False
model_parameters.initial_lr     = 0.03

# Running on CPU
if len(physical_devices)==0:
    model_parameters.batch_size     = 64
    model_parameters.output_var_dirs= 1
    model_parameters.stack_rnn_size = 1

# Get the necessary data
train_data = tf.data.Dataset.from_tensor_slices(training_data)
val_data   = tf.data.Dataset.from_tensor_slices(validation_data)
test_data  = tf.data.Dataset.from_tensor_slices(test_data)

# Form batches
batched_train_data = train_data.batch(model_parameters.batch_size)
batched_val_data   = val_data.batch(model_parameters.batch_size)
batched_test_data  = test_data.batch(model_parameters.batch_size)

# Model
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

# Quantitative testing: ADE/FDE
print("[INF] Quantitative testing")
results = evaluation_minadefde(tj_enc_dec,batched_test_data,model_parameters)
plot_comparisons_minadefde(results,dataset_names[idTest])
print(results)

# Qualitative testing
qualitative = True
if qualitative==True:
    print("[INF] Qualitative testing")
    for i in range(5):
        batch, test_bckgd = get_testing_batch(test_data,dataset_dir+dataset_names[idTest])
        evaluation_qualitative(tj_enc_dec,batch,model_parameters,background=test_bckgd,homography=test_homography, flip=False,n_peds_max=1,display_mode=None)
