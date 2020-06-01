import os
import pickle
import numpy as np
import random

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
            # TODO: (Improve) Store only the (x,y) coordinates for now
            data.append(traj[[0, 1], :].T)
            # Number of batches this datapoint is worth
            counter += int(traj.shape[1] / ((seq_length_obs+1)))

    # Calculate the number of batches (each of batch_size) in the data
    #counter tiene la cantidad de bloques de 8 pasos
    num_batches = int(counter /batch_size)
    #cada bache tiene batch_size conjuntos donde cada conjunto tiene datos de length+2
    return data,num_batches
