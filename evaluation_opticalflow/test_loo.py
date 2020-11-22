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
from training_and_testing import Trainer,Tester,Experiment_Parameters
import matplotlib.pyplot as plt
from model import TrajectoryEncoderDecoder, Model_Parameters
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
import random
from datetime import datetime
random.seed(datetime.now())
from traj_utils import relative_to_abs, vw_to_abs
from datasets_utils import setup_loo_experiment

if tf.test.gpu_device_name():
    print('[INF] Using GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("[INF] Using CPU")

# Load the default parameters
experiment_parameters = Experiment_Parameters(add_social=False,add_kp=False,obstacles=False)
#experiment_parameters.output_representation = 'vw'

dataset_dir       = "../data1/"
dataset_paths     = [dataset_dir+'eth-hotel',dataset_dir+'eth-univ',dataset_dir+'ucy-zara01',dataset_dir+'ucy-zara02',dataset_dir+'ucy-univ']

# Load the dataset and perform the split
training_data,validation_data,test_data,test_bckgd,test_homography = setup_loo_experiment('ETH_UCY',dataset_paths,0,experiment_parameters)

# Plot ramdomly a subset of the training data (spatial data only)
show_training_samples = False
if show_training_samples:
    training = len(training_data[list(training_data.keys())[0]])
    nSamples = min(20,training)
    samples  = random.sample(range(1,training), nSamples)
    plt.subplots(1,1,figsize=(10,10))
    plt.subplot(1,1,1)
    plt.axis('equal')
    # Plot some of the training data
    for (o,t,p,r) in zip(training_data["obs_traj"][samples],training_data["obs_traj_theta"][samples],training_data["pred_traj"][samples],training_data["pred_traj_rel"][samples]):
        # Observations
        plt.plot(o[:,0],o[:,1],color='red')
        # From the last observed point to the first target
        plt.plot([o[-1,0],p[0,0]],[o[-1,1],p[0,1]],color='blue')
        plt.arrow(o[-1,0], o[-1,1], 0.5*math.cos(t[-1,0]),0.5*math.sin(t[-1,0]), head_width=0.05, head_length=0.1, fc='k', ec='k')
        # Prediction targets
        plt.plot(p[:,0],p[:,1],color='blue',linewidth=3)
        if experiment_parameters.output_representation == 'vw':
            pred_vw = vw_to_abs(r, o[-1])
        else:
            pred_vw = relative_to_abs(r, o[-1])
        plt.plot(pred_vw[:,0],pred_vw[:,1],color='yellow',linewidth=1)

    plt.show()

#############################################################
# Model parameters
model_parameters = Model_Parameters(add_attention=True,add_kp=experiment_parameters.add_kp,add_social=experiment_parameters.add_social,output_representation=experiment_parameters.output_representation)
if experiment_parameters.output_representation == 'vw':
    model_parameters.num_epochs = 100
    model_parameters.initial_lr = 0.1

#model_parameters.num_epochs     = 3

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
plot_training    = False
if perform_training==True:
        # TODO: Use ttf.data.Dataset!
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data.data["obs_traj_rel"],train_data.data["pred_traj_rel"]))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(512)

        train_loss_results,val_loss_results,val_metrics_results,__ = tj_enc_dec.training_loop(train_data,val_data,model_parameters,checkpoint,checkpoint_prefix)
        if plot_training==True:
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
        tj_enc_dec.qualitative_evaluation(test_data,model_parameters,10,background=test_bckgd,homography=test_homography)
