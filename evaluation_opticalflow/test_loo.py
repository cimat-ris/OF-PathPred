# Imports
import sys,os
''' TF_CPP_MIN_LOG_LEVEL
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append('./lib')
import math,numpy as np
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
print('[INF] Tensorflow version: ',tf.__version__)
tf.test.gpu_device_name()
# Important imports
import batches_data
from training_and_testing import Experiment_Parameters
import matplotlib.pyplot as plt
from model import TrajectoryEncoderDecoder, Model_Parameters
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from datasets_utils import setup_loo_experiment, get_testing_batch
from plot_utils import plot_training_data,plot_training_results

if tf.test.gpu_device_name():
    print('[INF] Using GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("[INF] Using CPU")

# Load the default parameters
experiment_parameters = Experiment_Parameters(add_social=True,add_kp=False,obstacles=False)
#experiment_parameters.output_representation = 'vw'

dataset_dir       = "../datasets/"
dataset_paths     = [dataset_dir+'eth-hotel',dataset_dir+'eth-univ',dataset_dir+'ucy-zara01',dataset_dir+'ucy-zara02',dataset_dir+'ucy-univ']

# Load the dataset and perform the split
idTest = 2
training_data,validation_data,test_data,test_homography = setup_loo_experiment('ETH_UCY',dataset_paths,idTest,experiment_parameters,use_pickled_data=False)

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
model_parameters.num_epochs      = 35
model_parameters.output_var_dirs = 3
model_parameters.batch_size      = 128

# Get the necessary data
# TODO: replace these structures by the tf ones
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
                                decoder=tj_enc_dec.dec,
                                enctodec=tj_enc_dec.enctodec,
                                ft_class=tj_enc_dec.ft_class,
                                ot_class=tj_enc_dec.ot_class)

# Training
print("[INF] Training the model")
perform_training = True
plot_training    = True
if perform_training==True:
    # TODO: Use tf.data.Dataset!
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data.data["obs_traj_rel"],train_data.data["pred_traj_rel"]))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(512)
    train_loss_results,val_loss_results,val_metrics_results,__ = tj_enc_dec.training_loop(train_data,val_data,model_parameters,checkpoint,checkpoint_prefix)
    if plot_training==True:
        plot_training_results(train_loss_results,val_loss_results,val_metrics_results)

# Testing
# Restoring the latest checkpoint in checkpoint_dir
print("[INF] Restoring last model")
status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Quantitative testing: ADE/FDE
print("[INF] Quantitative testing")
results = tj_enc_dec.quantitative_evaluation(test_data,model_parameters)
print(results)

# Qualitative testing
qualitative = True
if qualitative==True:
    print("[INF] Qualitative testing")
    for i in range(10):
        batch, test_bckgd = get_testing_batch(test_data,dataset_paths[idTest],model_parameters)
        tj_enc_dec.qualitative_evaluation(batch,model_parameters,background=test_bckgd,homography=test_homography, flip=False)
