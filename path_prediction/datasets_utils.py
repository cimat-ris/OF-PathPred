import os
import pickle
import numpy as np
import matplotlib.image as mpimg
import numpy as np
import cv2
import tensorflow as tf
from path_prediction.process_file import process_file
from path_prediction.batches_data import get_batch

def get_testing_batch(testing_data,testing_data_path):
    # A trajectory id
    testing_data_arr = list(testing_data.as_numpy_iterator())
    randomtrajId     = np.random.randint(len(testing_data_arr),size=1)[0]
    frame_id         = testing_data_arr[randomtrajId]["frames_ids"][0]
    # Get the video corresponding to the testing
    cap   = cv2.VideoCapture(testing_data_path+'/video.avi')
    frame = 0
    while(cap.isOpened()):
        ret, test_bckgd = cap.read()
        if frame == frame_id:
            break
        frame = frame + 1
    # Form the batch
    filtered_data  = testing_data.filter(lambda x: x["frames_ids"][0]==frame_id)
    filtered_data  = filtered_data.batch(20)
    for element in filtered_data.as_numpy_iterator():
        return element, test_bckgd

def setup_loo_experiment(experiment_name,ds_path,ds_names,leave_id,experiment_parameters,use_pickled_data=False,pickle_dir='pickle/',validation_proportion=0.1):
    # Dataset to be tested
    testing_datasets_names  = [ds_names[leave_id]]
    training_datasets_names = ds_names[:leave_id]+ds_names[leave_id+1:]
    print('[INF] Testing/validation dataset:',testing_datasets_names)
    print('[INF] Training datasets:',training_datasets_names)
    if not use_pickled_data:
        # Process data specified by the path to get the trajectories with
        print('[INF] Extracting data from the datasets')
        test_data  = process_file(ds_path, testing_datasets_names, experiment_parameters)
        train_data = process_file(ds_path, training_datasets_names, experiment_parameters)

        # Count how many data we have (sub-sequences of length 8, in pred_traj)
        n_test_data  = len(test_data[list(test_data.keys())[2]])
        n_train_data = len(train_data[list(train_data.keys())[2]])
        idx          = np.random.permutation(n_train_data)
        # TODO: validation should be done from a similar distribution as test set!
        validation_pc= validation_proportion
        validation   = int(n_train_data*validation_pc)
        training     = int(n_train_data-validation)

        # Indices for training
        idx_train = idx[0:training]
        #  Indices for validation
        idx_val   = idx[training:]
        # Training set
        training_data = {
            "obs_traj":      train_data["obs_traj"][idx_train],
            "obs_traj_rel":  train_data["obs_traj_rel"][idx_train],
            "obs_traj_theta":train_data["obs_traj_theta"][idx_train],
            "pred_traj":     train_data["pred_traj"][idx_train],
            "pred_traj_rel": train_data["pred_traj_rel"][idx_train],
            "frames_ids":    train_data["frames_ids"][idx_train]
        }
        if experiment_parameters.add_social:
            training_data["obs_optical_flow"]=train_data["obs_optical_flow"][idx_train]
        # Test set
        testing_data = {
            "obs_traj":      test_data["obs_traj"][:],
            "obs_traj_rel":  test_data["obs_traj_rel"][:],
            "obs_traj_theta":test_data["obs_traj_theta"][:],
            "pred_traj":     test_data["pred_traj"][:],
            "pred_traj_rel": test_data["pred_traj_rel"][:],
            "frames_ids":    test_data["frames_ids"][:]
        }
        if experiment_parameters.add_social:
            testing_data["obs_optical_flow"]=test_data["obs_optical_flow"][:]
        # Validation set
        validation_data ={
            "obs_traj":      train_data["obs_traj"][idx_val],
            "obs_traj_rel":  train_data["obs_traj_rel"][idx_val],
            "obs_traj_theta":train_data["obs_traj_theta"][idx_val],
            "pred_traj":     train_data["pred_traj"][idx_val],
            "pred_traj_rel": train_data["pred_traj_rel"][idx_val],
            "frames_ids":    train_data["frames_ids"][idx_val]
        }
        if experiment_parameters.add_social:
            validation_data["obs_optical_flow"]=train_data["obs_optical_flow"][idx_val]

        # Training dataset
        pickle_out = open(pickle_dir+'/training_data_'+experiment_name+'.pickle',"wb")
        pickle.dump(training_data, pickle_out, protocol=2)
        pickle_out.close()

        # Test dataset
        pickle_out = open(pickle_dir+'/test_data_'+experiment_name+'.pickle',"wb")
        pickle.dump(test_data, pickle_out, protocol=2)
        pickle_out.close()

        # Validation dataset
        pickle_out = open(pickle_dir+'/validation_data_'+experiment_name+'.pickle',"wb")
        pickle.dump(validation_data, pickle_out, protocol=2)
        pickle_out.close()
    else:
        # Unpickle the ready-to-use datasets
        print("[INF] Unpickling...")
        pickle_in = open(pickle_dir+'/training_data_'+experiment_name+'.pickle',"rb")
        training_data = pickle.load(pickle_in)
        pickle_in = open(pickle_dir+'/test_data_'+experiment_name+'.pickle',"rb")
        test_data = pickle.load(pickle_in)
        pickle_in = open(pickle_dir+'/validation_data_'+experiment_name+'.pickle',"rb")
        validation_data = pickle.load(pickle_in)

    print("[INF] Training data: "+ str(len(training_data[list(training_data.keys())[0]])))
    print("[INF] Test data: "+ str(len(test_data[list(test_data.keys())[0]])))
    print("[INF] Validation data: "+ str(len(validation_data[list(validation_data.keys())[0]])))

    # Load the homography corresponding to this dataset
    homography_file = os.path.join(ds_path+testing_datasets_names[0]+'/H.txt')
    test_homography = np.genfromtxt(homography_file)
    return training_data,validation_data,test_data,test_homography
