
import numpy as np

# Divide a long sequence into mini-sequences of seq_length_obs+1 data (x,y)
def split_sequence_training_xy(seq_length_obs,data):
    length=int(len(data))
    X,Y=[],[]
    for j in range(length):
        traj = data[j]
        lon  = traj.shape[0]-seq_length_obs
        for i in range(0,lon):
            # Form sub-sequences of seq_length_obs data
            a = traj[i:(i +seq_length_obs ), :]
            X.append(a)
            # The target value is the next one (absolute values) in the sequence
            b = traj[i +seq_length_obs,:]
            Y.append(b)
    return np.array(X),np.array(Y)

# Divide a long sequence into mini-sequences of seq_length_obs+1 data (dx,dy)
def split_sequence_training_dxdy(seq_length_obs,data):
    length = int(len(data))
    X,Y = [],[]
    for j in range(length):
        traj = data[j]
        lon  = traj.shape[0]-seq_length_obs-1
        for i in range(0,lon+1):
            # Form sub-sequences of seq_length_obs data
            a = traj[i:(i +seq_length_obs ), :]
            X.append(a)
            # The target value is the increment to the next one
            b = traj[i+seq_length_obs, :]
            Y.append(b-a[len(a)-1,:])
    return np.array(X), np.array(Y)

# Compute the linear interpolation model
def linear_lsq_model(x,y):
    t      = range(1,len(x)+1)
    x_mean = np.mean(x)
    t_mean = np.mean(t)
    t_var  = np.var(t)
    xt_cov = np.cov (x, t)[0][1]
    vx     = xt_cov/t_var
    x0     = x_mean-(vx*t_mean)

    y_mean = np.mean(y)
    yt_cov = np.cov (y, t)[0][1]
    vy     = yt_cov/t_var
    y0     = y_mean-(vy*t_mean)
    return x0,y0,vx,vy

# Divide a long sequence into mini-sequences of seq_length_obs+1 data (deviations to the linear model)
def split_sequence_training_lineardev(seq_length_obs,data):
    length = int(len(data))
    X,Y = [],[]
    for j in range(length):
        traj = data[j]
        lon  = traj.shape[0]-seq_length_obs
        for i in range(0,lon):
            total = traj[i:(i + seq_length_obs ), :]
            X.append(total)
            xx = traj[i:(i + seq_length_obs ), 0]
            yy = traj[i:(i + seq_length_obs ), 1]

            # Compute the linear interpolation model
            # TODO: vary the support of the interpolation model
            x0,y0,vx,vy = linear_lsq_model(xx,yy)
            x_next      = x0+vx*(len(xx)+1)
            y_next      = y0+vy*(len(yy)+1)
            Y.append(traj[i+seq_length_obs, :]-[x_next,y_next])
    return np.array(X), np.array(Y)

# Prepare sequences for testing
def split_sequence_start_testing(seq_length_obs,data,seq_length_pred):
    tamano = int(len(data))
    X,Y_true = [],[]
    for j in range(tamano):
        traj = data[j]
        X.append(traj[0:seq_length_obs,:])
        Y_true.append(traj[0:seq_length_obs+seq_length_pred,:])
    return np.array(X),np.array(Y_true)

# This function takes a set of trajectories and build sub-sequences seq_length_obs+seq_length_pred
def split_sequence_testing(seq_length_obs,data,seq_length_pred):
    tamano = int(len(data))
    X,Y_true = [],[]
    # se recorre todo los datos de test
    for j in range(tamano):
        traj = data[j]
        lon = traj.shape[0]-seq_length_obs-seq_length_pred
        for i in range(0,lon+1):
            a = traj[i:(i +seq_length_obs ), :]
            X.append(a)
            # The full trajectory
            b = traj[i: (i+seq_length_obs+seq_length_pred), :]
            Y_true.append(b)
    return np.array(X),np.array(Y_true)
