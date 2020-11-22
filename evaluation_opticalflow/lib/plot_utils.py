from matplotlib import pyplot as plt
from obstacles import image_to_world_xy
import numpy as np

# Useful function to plot
def plot_gt_preds(traj_gt,traj_obs,traj_pred,pred_att_weights,background=None,homography=None):
    plt.subplots(1,1,figsize=(10,10))
    ax = plt.subplot(1,1,1)
    ax.set_title('Trajectory samples')
    plt.axis('equal')
    if background is not None:
        plt.imshow(background)
    # Plot some random testing data and the predicted ones
    plt.plot(traj_obs[0][0,0],traj_obs[0][0,1],color='red',label='Observations')
    plt.plot(traj_gt[0][0,0],traj_gt[0][0,1],color='blue',label='Ground truth')
    plt.plot(traj_pred[0][0,0],traj_pred[0][0,1],color='green',label='Prediction')
    if homography is not None:
        homography = np.linalg.inv(homography)
    else:
        ax.set_ylabel('Y (m)')
        ax.set_xlabel('X (m)')        
    for (gt,obs,pred) in zip(traj_gt,traj_obs,traj_pred):
        if homography is not None:
            gt  = image_to_world_xy(gt, homography)
            obs = image_to_world_xy(obs, homography)
            pred= image_to_world_xy(pred, homography)
        plt.plot(obs[:,1],obs[:,0],color='red')
        # Ground truth trajectory
        plt.plot([obs[-1,1],gt[0,1]],[obs[-1,0],gt[0,0]],color='blue')
        plt.plot(gt[:,1],gt[:,0],color='blue')
        # Predicted trajectory
        plt.plot([obs[-1,1],pred[0,1]],[obs[-1,0],pred[0,0]],color='green')
        plt.plot(pred[:,1],pred[:,0],color='green')
    ax.legend()
    plt.show()
