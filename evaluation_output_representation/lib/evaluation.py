import numpy as np
import matplotlib.pyplot as plt

# Evaluate the ADE
def evaluate_ade(predicted_traj, true_traj, observed_length):
    error = np.zeros(len(true_traj) - observed_length)
    # For each point in the predicted trajectory
    for i in range(observed_length, len(true_traj)):
        # The predicted position
        pred_pos = predicted_traj[i]
        # The true position
        true_pos = true_traj[i]
        # The euclidean distance is the error
        error[i-observed_length] = np.linalg.norm(true_pos - pred_pos)

    # Return the mean error
    return np.mean(error)

# Evaluate the FDE
def evaluate_fde(predicted_traj, true_traj, observed_length):
    return np.linalg.norm(predicted_traj[-1]-true_traj[-1])

def plot_qualitative(p,v,name):
    color_names = ["r", "g", "b","c","m","y", "peachpuff","grey", "fuchsia","violet",
                   "teal","seagreen","lime","yellow","coral","aquamarine","hotpink"]
    plt.plot(p[0][0:8,0],p[0][0:8,1],'*--',color= color_names[2],label = 'Observed')
    plt.plot(p[0][7:,0],p[0][7:,1],'-',color=color_names[2],label='Predicted')
    plt.plot(v[0][7:,0],v[0][7:,1],'--',color=color_names[4],label='Ground truth')

    plt.plot(p[1][0:8,0],p[1][0:8,1],'*--',color= color_names[2])
    plt.plot(p[1][7:,0],p[1][7:,1],'-',color=color_names[2])
    plt.plot(v[1][7:,0],v[1][7:,1],'--',color=color_names[4])

    plt.legend()
    plt.title('Prediction sample {}'.format(name))
    plt.savefig(name)
    plt.show()
