import numpy as np
import matplotlib.pyplot as plt
from lib.sequence_preparation import *

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

# Takes a testing set and evaluates errors on it
def evaluate_testing_set(model,datos, seq_length_obs, seq_length_pred,pixels=False, mode='xy'):
    # Observations and targets, by splitting the trajectories of the testing set
    X,Y_true  = split_sequence_testing(seq_length_obs,datos,seq_length_pred)
    total_ade = 0.0
    total_fde = 0.0
    # Iterate over the splitted testing data
    for i in range(len(X)):
        traj_obs  = X[i]
        traj_pred = X[i]
        # For each timestep to predict
        for j in range(seq_length_pred):
            traj_obsr = np.reshape(traj_obs, (1,traj_obs.shape[0],traj_obs.shape[1]) )
            # Applies the model to produce the next point
            next_point= model.predict(traj_obsr)
            if mode=='dxdy':
                next_point=next_point+traj_obs[-1]
            if mode=='lineardev':
                x0,y0,vx,vy   = linear_lsq_model(traj_obs[:,0],traj_obs[:,1])
                x_pred_linear = x0+vx*(len(traj_obs[:,0])+1)
                y_pred_linear = y0+vy*(len(traj_obs[:,1])+1)
                next_point=next_point+[x_pred_linear,y_pred_linear]
            traj_obs  = np.concatenate((traj_obs[1:len(traj_obs)],next_point),axis=0)
            traj_pred = np.concatenate((traj_pred,next_point),axis=0)
        # Evaluate the difference
        if pixels:
            traj_pred_pixels = np.column_stack((768*traj_pred[:,0],576*traj_pred[:,1]))
            traj_gt_pixels   = np.column_stack((768*Y_true[i][:,0],576*Y_true[i][:,1]))
            total_ade += evaluate_ade(traj_pred_pixels, traj_gt_pixels, seq_length_obs)
            total_fde += evaluate_fde(traj_pred_pixels, traj_gt_pixels, seq_length_obs)
        else:
            total_ade += evaluate_ade(traj_pred, Y_true[i], seq_length_obs)
            total_fde += evaluate_fde(traj_pred, Y_true[i], seq_length_obs)

    error_ade = total_ade/len(X)
    error_fde = total_fde/len(X)
    if pixels:
        print('---------Error (pixels)--------')
    else:
        print('---------Error (normalized coordinates)--------')
    print('[RES] ADE: ',error_ade)
    print('[RES] FDE: ',error_fde)

# Quantitative evaluation
def evaluate_testing_set_start(model,datos, seq_length_obs, seq_length_pred, mode='xy'):
    # Just the starting parts of the sequence
    X,Y_true    = split_sequence_start_testing(seq_length_obs,datos,seq_length_pred)
    total_error = 0.0
    plt.figure(figsize=(18,15))
    color_names       = ["r","crimson" ,"g", "b","c","m","y","lightcoral", "peachpuff","grey","springgreen" ,"fuchsia","violet","teal","seagreen","lime","yellow","coral","aquamarine","hotpink"]
    plt.subplot(1,1,1)
    print(len(X))
    # For all the subsequences
    for i in range(len(X)):
        traj_obs  = X[i]
        traj_pred = X[i]
        for j in range(seq_length_pred):
            # Observation reshaped to be fed to the model
            traj_obsr  = np.reshape(traj_obs, (1,traj_obs.shape[0],traj_obs.shape[1]) )
            # Next position predicted
            next_point = model.predict(traj_obsr)
            if mode=='dxdy':
                next_point=next_point+traj_obs[-1]
            if mode=='lineardev':
                x0,y0,vx,vy   = linear_lsq_model(traj_obs[:,0],traj_obs[:,1])
                x_pred_linear = x0+vx*(len(traj_obs[:,0])+1)
                y_pred_linear = y0+vy*(len(traj_obs[:,1])+1)
                next_point=next_point+[x_pred_linear,y_pred_linear]
            # Next observation will be shifted by one to the right
            traj_obs  = np.concatenate((traj_obs[1:len(traj_obs)],next_point),axis=0)
            traj_pred = np.concatenate((traj_pred,next_point),axis=0)

        error_ade = evaluate_ade(traj_pred,Y_true[i],seq_length_obs)
        error_fde = evaluate_fde(traj_pred,Y_true[i],seq_length_obs)

        plt.plot(Y_true[i][0:8,0],Y_true[i][0:8,1],'*--',color=color_names[19-i],label = 'Observed')
        plt.plot(Y_true[i][7:,0],Y_true[i][7:,1],'--',color=color_names[i],label = 'GT')
        plt.plot(traj_pred[seq_length_obs-1:,0],traj_pred[seq_length_obs-1:,1],'-',color=color_names[19-i],label = 'Predicted')
        plt.axis('equal')
        total_error += error_ade
    plt.title("Predicting 4 positions with LTM-X-Y")
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    error_model = total_error/len(X)
    print("[RES] Average error ",error_model)
    #plt.savefig("4predichas.pdf")
    plt.show()


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
    plt.savefig(name)
    plt.show()

# Esta funcion hace la prediccion en coordenadas pixel
def sample_en_pixeles_cualitativamente(model, datos, seq_length_obs, seq_length_pred, mode='xy'):


    X,Y_true = split_sequence_testing(seq_length_obs,datos,seq_length_pred)
    total_error = 0.0
    total_final = 0.0
    trayectoria = []
    verdadero= []

    for i in range(len(X)):
        traj_obs = X[i]
        traj_pred = X[i]

        for j in range(seq_length_pred):
            traj_obsr  = np.reshape(traj_obs, (1,traj_obs.shape[0],traj_obs.shape[1]) )
            # Applies the model
            next_point = model.predict(traj_obsr)
            if mode=='dxdy':
                next_point=next_point+traj_obs[-1]
            if mode=='lineardev':
                x0,y0,vx,vy   = linear_lsq_model(traj_obs[:,0],traj_obs[:,1])
                x_pred_linear = x0+vx*(len(traj_obs[:,0])+1)
                y_pred_linear = y0+vy*(len(traj_obs[:,1])+1)
                next_point=next_point+[x_pred_linear,y_pred_linear]
            traj_obs   = np.concatenate((traj_obs[1:len(traj_obs)], next_point), axis = 0)
            traj_pred  = np.concatenate((traj_pred, next_point), axis = 0)

        traj_pre =  np.column_stack((768*traj_pred[:,0],576*traj_pred[:,1]))
        traj_tr = np.column_stack((768*Y_true[i][:,0],576*Y_true[i][:,1]))

        trayectoria.append(traj_pre)
        verdadero.append(traj_tr)
        #SE CALCULA LA METRICA ADE

        total_error += evaluate_ade(traj_pre, traj_tr, seq_length_obs)
        total_final += evaluate_fde(traj_pre, traj_tr, seq_length_obs)

    error_modelo     = total_error/len(X)
    error_fde_modelo = total_final/len(X)

    print('---------Error--------')
    print('ADE')
    print(error_modelo)
    print('FDE')
    print(error_fde_modelo)
    return trayectoria, verdadero
