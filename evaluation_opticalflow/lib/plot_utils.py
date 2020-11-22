from matplotlib import pyplot as plt


# Useful function to plot
def plot_gt_preds(traj_gt,traj_obs,traj_pred,pred_att_weights):
    plt.subplots(1,1,figsize=(10,10))
    ax = plt.subplot(1,1,1)
    ax.set_ylabel('Y (m)')
    ax.set_xlabel('X (m)')
    ax.set_title('Trajectory samples')
    plt.axis('equal')
    # Plot some random testing data and the predicted ones
    plt.plot(traj_obs[0][0,0],traj_obs[0][0,1],color='red',label='Observations')
    plt.plot(traj_gt[0][0,0],traj_gt[0][0,1],color='blue',label='Ground truth')
    plt.plot(traj_pred[0][0,0],traj_pred[0][0,1],color='green',label='Prediction')
    for (gt,obs,pred) in zip(traj_gt,traj_obs,traj_pred):
        plt.plot(obs[:,0],obs[:,1],color='red')
        # Ground truth trajectory
        plt.plot([obs[-1,0],gt[0,0]],[obs[-1,1],gt[0,1]],color='blue')
        plt.plot(gt[:,0],gt[:,1],color='blue')
        # Predicted trajectory
        plt.plot([obs[-1,0],pred[0,0]],[obs[-1,1],pred[0,1]],color='green')
        plt.plot(pred[:,0],pred[:,1],color='green')
    ax.legend()
    plt.show()
