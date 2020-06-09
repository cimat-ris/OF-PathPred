
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
def linear_lsq_model(x,y,length_r):
    l_obs  = len(x)

    t      = range(1,len(x)+1)
    if(length_r<l_obs):
        t = t[obs-longitud:obs]
        x = x[obs-longitud:obs]
        y = y[obs-longitud:obs]
    
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
def split_sequence_training_lineardev(seq_length_obs,data,length_r):
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
            
            x0,y0,vx,vy = linear_lsq_model(xx,yy,length_r)
            x_next      = x0+vx*(len(xx)+1)
            y_next      = y0+vy*(len(yy)+1)
            Y.append(traj[i+seq_length_obs, :]-[x_next,y_next])
    return np.array(X), np.array(Y)

# Divide a long sequence into mini-sequences of seq_length_obs+1 data (model of only displacement)
def split_sequence_training_only_displacement(seq_length_obs,data):
 
    length = int(len(data))
    
    X,Y = [],[]
    for j in range(length):
        traj = data[j]
        lon  = traj.shape[0]-seq_length_obs
        for i in range(0,lon):
            pos_rel = np.zeros((seq_length_obs+1,2), dtype="float")
            pos_abs = traj[i:(i +seq_length_obs+1 ), :]
            pos_rel[1:, :] = pos_abs[1:, :] - pos_abs[:-1, :]
            X.append(pos_rel[:seq_length_obs,:]) 
            Y.append(pos_rel[seq_length_obs:])
    Y = np.concatenate(Y,axis=0)
    return np.array(X), Y

# Prepare sequences for testing the models of output representation
def split_sequence_start_testing_ouput_representation(data,seq_length_obs,seq_length_pred):
    tamano = int(len(data))
    X,Y_true = [],[]
    for j in range(tamano):
        traj = data[j]
        X.append(traj[0:seq_length_obs,:])
        Y_true.append(traj[0:seq_length_obs+seq_length_pred,:])
    return np.array(X),np.array(Y_true)

# Prepare sequences for testing the model only displacement
def split_sequence_start_testing_only_displacement(data,seq_length_obs,seq_length_pred):
    tamano = int(len(data))
    X,Y_true = [],[]
    for j in range(tamano):
        traj    = data[j]
        pos_rel = np.zeros((seq_length_obs,2), dtype="float")
        pos_abs = traj[0:seq_length_obs, :]
        pos_rel[1:,:] = pos_abs[1:, :] - pos_abs[:-1, :]
        X.append(pos_rel)
        Y_true.append(traj[0:seq_length_obs+seq_length_pred,:])
    return np.array(X),np.array(Y_true)


def split_sequence_start_testing(data,seq_length_obs,seq_length_pred,representation_mode):
    if representation_mode=='only_displacement':
        allX,allY = split_sequence_start_testing_only_displacement(data,seq_length_obs,seq_length_pred) 
    else:
        allX,allY = split_sequence_start_testing_ouput_representation(data,seq_length_obs,seq_length_pred)
    return allX,allY  
        
# This function takes a set of trajectories (test) and build sub-sequences seq_length_obs+seq_length_pred
def split_sequence_testing_output_representation(data,seq_length_obs,seq_length_pred):
    tamano = int(len(data))
    X,Y_true = [],[]
    # se recorre todo los datos de test
    for j in range(tamano):
        traj = data[j]
        lon  = traj.shape[0]-seq_length_obs-seq_length_pred
        for i in range(0,lon+1):
            a = traj[i:(i +seq_length_obs ), :]
            X.append(a)
            # The full trajectory
            b = traj[i: (i+seq_length_obs+seq_length_pred), :]
            Y_true.append(b)
    return np.array(X),np.array(Y_true)

# This fuction takes a set of trajectories (test) and build sub-sequences seq_length_obs+seq_length_pred
# to the model of only displacement
def split_sequence_testing_only_displacement(data,seq_length_obs,seq_length_pred):
    tamano = int(len(data))
    
    X_des,Y_abs = [],[]
    for j in range(tamano):
        traj = data[j]
        lon = traj.shape[0]-seq_length_obs-seq_length_pred
        for i in range(0,lon+1):
            pos_rel = np.zeros((seq_length_obs+seq_length_pred,2), dtype='float')
            pos_abs = traj[i:(i +seq_length_obs+ seq_length_pred ), :]
            
            pos_rel[1:,:] = pos_abs[1:,:]-pos_abs[:-1,:]
            
            X_des.append(pos_rel[:seq_length_obs,:])
      
            Y_abs.append(pos_abs[0:seq_length_obs+seq_length_pred ,:])
            
    return np.array(X_des),np.array(Y_abs)

def split_sequence_testing(data,seq_length_obs,seq_length_pred, representation_mode):
    if representation_mode=='only_displacement':
        X,Y = split_sequence_testing_only_displacement(data,seq_length_obs,seq_length_pred)
    else: 
        X,Y = split_sequence_testing_output_representation(data,seq_length_obs,seq_length_pred)
    return X,Y    

        
        
