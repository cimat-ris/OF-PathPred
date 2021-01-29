from matplotlib import pyplot as plt
from obstacles import image_to_world_xy
from traj_utils import relative_to_abs, vw_to_abs
import numpy as np
import random
import math
import cv2
import tensorflow as tf
from datetime import datetime
import seaborn as sns
random.seed(datetime.now())

# Visualization of the training data
def plot_training_data(training_data,experiment_parameters):
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

# Visualization of the training results
def plot_training_results(train_loss_results,val_loss_results,val_metrics_results):
    # Plot training results
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(train_loss_results,'b',label='Training')
    ax.plot(val_loss_results,'r',label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and validation losses')
    ax.legend()
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(val_metrics_results["mADE"],'b',label='mADE in validation')
    ax.plot(val_metrics_results["mFDE"],'r',label='mFDE in validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error (m)')
    ax.set_title('Metrics at validation')
    ax.legend()
    plt.show()


# Visualization of the predictions vs. the ground truth
def plot_gt_preds(traj_gt,traj_obs,traj_pred,neighbors_gt,distributions_pred,background=None,homography=None,flip=False,display_mode=None):
    plt.subplots(1,1,figsize=(10,10))
    ax = plt.subplot(1,1,1)
    ax.set_title('Trajectory samples')
    plt.axis('equal')
    if background is not None:
        plt.imshow(background)
    # Get the number of samples per prediction
    nMCSamples  = traj_pred[0].shape[0]
    nModeSamples= traj_pred[0].shape[1]
    # Plot some random testing data and the predicted ones
    plt.plot([0],[0],color='purple',label='Neighbors')
    plt.plot([0],[0],color='red',   label='Observations')
    plt.plot([0],[0],color='blue',  label='Ground truth')
    plt.plot([0],[0],color='green', label='Prediction')
    if homography is not None:
        homography = np.linalg.inv(homography)
    else:
        ax.set_ylabel('Y (m)')
        ax.set_xlabel('X (m)')
    for i,(gt,obs,neighbors) in enumerate(zip(traj_gt,traj_obs,neighbors_gt)):
        neighbors = neighbors[0,:,1:3]
        neighbors = np.array([x for x in neighbors[:] if abs(x[0])>0.001])
        mc_preds  = traj_pred[i]
        if homography is not None:
            gt       = image_to_world_xy(gt, homography,flip=flip)
            obs      = image_to_world_xy(obs, homography,flip=flip)
            if neighbors.shape[0]>0:
                neighbors= image_to_world_xy(neighbors, homography,flip=flip)
            tpred= image_to_world_xy(tf.reshape(mc_preds,[mc_preds.shape[0]*mc_preds.shape[1]*mc_preds.shape[2],mc_preds.shape[3]]), homography,flip=flip)
            mc_preds = tf.reshape(tpred,[mc_preds.shape[0],mc_preds.shape[1],mc_preds.shape[2],mc_preds.shape[3]])

        # Observed trajectory
        plt.plot(obs[:,0],obs[:,1],color='red')
        if neighbors.shape[0]>0:
            plt.plot(neighbors[:,0],neighbors[:,1],color='purple',marker='o',markersize=12,linestyle='None')
        # Predicted trajectories.
        # Scanning the MC samples
        for pred in mc_preds:
            for k in range(nModeSamples):
                if display_mode is not None and k!=display_mode:
                    continue
                plt.plot([obs[-1,0],pred[k][0,0]],[obs[-1,1],pred[k][0,1]],color='green')
                plt.plot(pred[k][:,0],pred[k][:,1],color='green')
                plt.text(pred[k][-1,0]+10*(pred[k][-1,0]-pred[k][-2,0])/tf.norm(pred[k][-1,0]-pred[k][-2,0]),pred[k][-1,1]+10*(pred[k][-1,1]-pred[k][-2,1])/tf.norm(pred[k][-1,1]-pred[k][-2,1]),"{}{}".format((k+1)//2,'+' if k%2==1 else '-'))
        # Ground truth trajectory
        plt.plot([obs[-1,0],gt[0,0]],[obs[-1,1],gt[0,1]],color='blue',linewidth=2)
        plt.plot(gt[:,0],gt[:,1],color='blue',linewidth=2)
    ax.legend()
    plt.show()
