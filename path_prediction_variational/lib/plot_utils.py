from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
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
    plt.savefig("figures_train.pdf")
    #plt.show()

# Visualization of the background
def plot_background(ax,background):
    ax.imshow(background)

# Visualization of neighbors
def plot_neighbors(ax,neighbors_gt,homography=None,flip=False):
    if homography is not None:
        homography = np.linalg.inv(homography)
    for i,neighbors in enumerate(neighbors_gt):
        neighbors = neighbors[0,:,1:3]
        neighbors = np.array([x for x in neighbors[:] if abs(x[0])>0.001])
        if homography is not None:
            if neighbors.shape[0]>0:
                neighbors= image_to_world_xy(neighbors, homography,flip=flip)
        if neighbors.shape[0]>0:
            ax.plot(neighbors[:,0],neighbors[:,1],color='purple',marker='o',markersize=12,linestyle='None')

# Visualization of the predictions vs. the ground truth
def plot_gt_preds(ax,traj_gt,traj_obs,traj_pred,homography=None,flip=False,display_mode=None,n_peds_max=1000,mode=0):
    ax.set_title('Trajectory samples with cSG-MCMC')
    #ax.axis('equal')
    # Get the number of samples per prediction
    nModeSamples= traj_pred[0].shape[0]
    # Plot some random testing data and the predicted ones
    ax.plot([0],[0],color='purple',label='Neighbors')
    ax.plot([0],[0],color='red',   label='Observations')
    ax.plot([0],[0],color='blue',  label='Ground truth')
    ax.plot([0],[0],color='green', label='Prediction')
    if homography is not None:
        homography = np.linalg.inv(homography)
    else:
        ax.set_ylabel('Y (m)')
        ax.set_xlabel('X (m)')
    
    for i,(gt,obs) in enumerate(zip(traj_gt,traj_obs)): # recorremos el batch
        if i>=n_peds_max: # para mas personas del batch
            break
        preds     = traj_pred[i]
        if (preds.shape[0]==0):
            continue
        if homography is not None:
            gt    = image_to_world_xy(gt, homography,flip=flip)
            obs   = image_to_world_xy(obs,homography,flip=flip)
            tpred = image_to_world_xy(tf.reshape(preds,[preds.shape[0]*preds.shape[1],preds.shape[2]]), homography,flip=flip)
            preds = tf.reshape(tpred,[preds.shape[0],preds.shape[1],preds.shape[2]])
        # Observed trajectory
        ax.plot(obs[:,0],obs[:,1],color='red')
        
        # Predicted trajectories.
        for k in range(nModeSamples):
            if display_mode is not None and k!=display_mode:
                continue
            ax.plot([obs[-1,0],preds[k][0,0]],[obs[-1,1],preds[k][0,1]],color=colores[mode],alpha=1.0)
            ax.plot(preds[k][:,0],preds[k][:,1],color=colores[mode],alpha=1.0)
            
        # Ground truth trajectory
        ax.plot([obs[-1,0],gt[0,0]],[obs[-1,1],gt[0,1]],color='blue',linewidth=2)
        ax.plot(gt[:,0],gt[:,1],color='blue',linewidth=2)

    custom_lines = [Line2D([0],[0],color="purple",marker="s", markersize=7, linestyle=""),
                    Line2D([0],[0],color="red",lw=2),
                    Line2D([0],[0],color="blue",lw=2),
                    Line2D([0],[0],color="green",lw=2),
                    
                    Line2D([0],[0],color="yellow",lw=2),
                    Line2D([0],[0],color="pink",lw=2),
                    Line2D([0],[0],color="purple",lw=2),
                    Line2D([0],[0],color="cyan",lw=2),
                    Line2D([0],[0],color="orange",lw=2),
                    Line2D([0],[0],color="olive",lw=2),
                    Line2D([0],[0],color="brown",lw=2)
                   ]
                   
    legends = ['Prediction C'+str(ciclo+1) for ciclo in range(mode+1)]
    ax.legend(custom_lines,['Neighbors','Observations','Ground truth'] + legends, loc='upper right')
    ax.axis("off")

# Visualization of the attention coefficients
def plot_attention(ax,traj_obs,traj_pred,attention,homography=None,flip=False,step=0):
    attention=attention[0]
    ax.set_title('Attention at step {}'.format(step))    
    ax.set_ylabel('Attention weight')
    ax.set_xlabel('Timestep in the observed trajectory')
    for i,obs in enumerate(traj_obs):
        if i>=1:
            break
        preds     = traj_pred[i]
        if (preds.shape[0]==0):
            continue
        for k in range(preds.shape[0]):
            legend = "Mode {}".format(k)
            ax.plot(attention[k][step],  label=legend)
    ax.legend()

colores = ["green",
           "yellow",
           "pink",
           "purple",
           "cyan",
           "orange",
           "olive",
           "brown",
           "grey",
           "black",
           "magenta",
           "white",
           "gold",
           "lightcoral",
           "darkred",
           "hotpink",
           "chocolate",
           "lightgray",
           "palegreen"
          ]




