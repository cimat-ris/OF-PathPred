import os
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
from lib.sequence_preparation import *


def plot_trajectories(data_p):
    ## Trajectory visualization
    import random
    color_names = ["r","crimson" ,"g", "b","c","m","y","lightcoral", "peachpuff","grey","springgreen" ,"fuchsia","violet","teal","seagreen","lime","yellow","coral","aquamarine","hotpink"]

    for i in range(len(data_p)):
        plt.plot(data_p[i][:,0],data_p[i][:,1],color=color_names[i])
    plt.title("Full trajectories in PETS-2009")
    plt.xlabel("x coordinate")
    plt.ylabel("y coordinate")
    plt.savefig("PETS2009-alltrajctories.pdf")
    plt.show()

# Preprocessing function
# Reads all the files in data_dirs/directory/data_filename
# Stores and outputs a tuple with:
# * a dictionary mapping each ped to their trajectories given by matrix 3 x numPoints with each column as x, y, frameId.
# * a list of the number of pedestrians for each data file
def preprocess(data_dirs,data_filename,output_file):
    # all_ped_data is a dictionary mapping each ped to their
    # trajectories given by matrix 3 x numPoints with each column
    # in the order x, y, frameId.
    # Pedestrians from all datasets are combined.
    # Dataset pedestrian indices are stored in dataset_indices.
    all_ped_data   ={}
    dataset_indices=[]
    current_ped    = 0
    # For each dataset
    for directory in data_dirs:
        # Define the path to its respective csv file
        file_path = os.path.join(directory,data_filename)

        # Load the data from the csv file.
        # Data are a 4xnumTrajPoints matrix.
        data = np.genfromtxt(file_path, delimiter=',')

        # Number of pedestrians in this dataset
        numPeds=np.size(np.unique(data[:,1]))
        print("[INF] Number of distinct pedestrians in"+ directory+": "+str(numPeds))

        # Iterate over the pedetrians
        for ped in range(1, numPeds+1):
            # Data for pedestrian ped
            traj = data[ data[:, 1] == ped]
            # Stored as (x,y,frame_Id)
            traj = traj[:, [2,3,0]]
            # Seen as [[x,...],[y,...],[Frame_Id,..]]
            traj=[list(traj[:,0]),list(traj[:,1]),list(traj[:,2])]
            all_ped_data[current_ped + ped] = np.array(traj)

        # Current dataset done
        dataset_indices.append(current_ped+numPeds)
        current_ped += numPeds

    # Whole data: t-uple of all the pedestrians data with the indices of the pedestrians
    complete_data = (all_ped_data, dataset_indices)
    # Stores all the data in a pickle file
    f = open(output_file, "wb")
    pickle.dump(complete_data, f, protocol=2)
    f.close()
    return complete_data


# Load the pickle file and filter the trajectories, and determine the number of batches they will generate.
def load_preprocessed(preprocessed_data_file,seq_length_obs,batch_size):

    # Load the pickle files
    f = open(preprocessed_data_file, "rb")
    raw_data = pickle.load(f)
    f.close()

    # Pedestrian data
    all_ped_data =raw_data[0]

    # We build  data as sequences with length seq_length_obs
    data    = []
    counter = 0

    # For each pedestrian data
    for ped in all_ped_data:

        # Extract trajectory of pedestrian ped
        traj = all_ped_data[ped]

        # If the length of the trajectory is greater than seq_length (+2 as we need both source and target data)
        #solo se toman las trajectorias de longitud mayor a seq_length+2
        if traj.shape[1] >= (seq_length_obs+1):
            data.append(traj[[0, 1], :].T)
            # Number of batches this datapoint is worth
            counter += int(traj.shape[1] / ((seq_length_obs+1)))

    # Calculate the number of batches (each of batch_size) in the data
    #counter tiene la cantidad de bloques de 8 pasos
    num_batches = int(counter /batch_size)
    #cada bache tiene batch_size conjuntos donde cada conjunto tiene datos de length+2
    return data,num_batches


# ## Splitting the data by ids
def split_data_idgroups(combination,intervals_ids,data):
    # Form different combinations of training/testing sets
    training_set = []
    for i in range(len(combination)-1):
        for j in intervals_ids[combination[i]]:
            training_set.append(data[j])
    testing_set = []
    for i in intervals_ids[combination[4]]:
        testing_set.append(data[i])
    return training_set,testing_set

def split_data(data_p,split_mode,length_obs,length_pred,representation_mode,id_train=1,length_r=8):
    total_length = len(data_p)

    if split_mode==0:
        indices      = range(total_length)
        intervals_ids        = []
        # Pedestrians are split in groups of size 20 percent of the number of pedestrians in the dataset
        step_size = int(np.ceil(total_length/5.))
        for i in range(0,total_length,step_size):
            intervals_ids.append(indices[i:i+step_size])
        # TODO: generalize?
        combinations=[(0,1,2,3,4),(0,1,2,4,3),(0,1,3,4,2),(0,2,3,4,1),(1,2,3,4,0)]
        # Generate train/test
        #for i in range(5):
        train1,test1 = split_data_idgroups(combinations[id_train-1],intervals_ids,data_p)

    elif split_mode==1:
        random.seed(0)
        indices      = np.arange(total_length)
        data_p = np.array(data_p)
        random.shuffle(indices)
        training_size = int(total_length * 0.80)
        testing_size  = total_length-training_size

        train1 = data_p[indices[0:training_size]]
        test1  = data_p[indices[training_size:]]

        print("[INF] Number of pedestrians "+str(total_length))
        print("[INF] Training with " ,len(train1))
        print("[INF] Testing with " ,len(test1))

    if representation_mode=='xy':
        trainX,trainY = split_sequence_training_xy(length_obs,train1)
    if representation_mode=='dxdy':
        trainX,trainY = split_sequence_training_dxdy(length_obs,train1)
    if representation_mode=='lineardev':
        trainX,trainY = split_sequence_training_lineardev(length_obs,train1,length_r)
    if representation_mode=='only_displacement':
        trainX,trainY = split_sequence_training_only_displacement(length_obs,train1)

    testX,testY = split_sequence_testing(test1,length_obs,length_pred,representation_mode)
    
    trainX       = np.reshape(trainX, (trainX.shape[0],trainX.shape[1],trainX.shape[2]))
    return trainX,trainY,testX,testY
