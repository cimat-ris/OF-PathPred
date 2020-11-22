import os
import pickle
import numpy as np
import matplotlib.image as mpimg
from process_file import process_file

def setup_loo_experiment(experiment_name,experiment_paths,leave_id,experiment_parameters,use_pickled_data=False,validation_proportion=0.1):
    if not use_pickled_data:
        # Dataset to be tested
        testing_data_paths        = [experiment_paths[leave_id]]
        training_data_paths       = experiment_paths[:leave_id]+experiment_paths[leave_id+1:]
        print('[INF] Testing dataset:',testing_data_paths)
        print('[INF] Training datasets:',training_data_paths)
        # Process data specified by the path to get the trajectories with
        print('[INF] Extracting data from the datasets')
        test_data  = process_file(testing_data_paths, experiment_parameters)
        train_data = process_file(training_data_paths, experiment_parameters)

        # Count how many data we have (sub-sequences of length 8, in pred_traj)
        n_test_data  = len(test_data[list(test_data.keys())[2]])
        n_train_data = len(train_data[list(train_data.keys())[2]])
        idx          = np.random.permutation(n_train_data)
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
        }
        if experiment_parameters.add_social:
            training_data["obs_flow"]=train_data["obs_flow"][idx_train]
        # Test set
        testing_data = {
            "obs_traj":     test_data["obs_traj"][:],
            "obs_traj_rel": test_data["obs_traj_rel"][:],
            "obs_traj_theta":test_data["obs_traj_theta"][:],
            "pred_traj":    test_data["pred_traj"][:],
            "pred_traj_rel":test_data["pred_traj_rel"][:],
        }
        if experiment_parameters.add_social:
            testing_data["obs_flow"]=test_data["obs_flow"][:]
        # Validation set
        validation_data ={
            "obs_traj":     train_data["obs_traj"][idx_val],
            "obs_traj_rel": train_data["obs_traj_rel"][idx_val],
            "obs_traj_theta":train_data["obs_traj_theta"][idx_val],
            "pred_traj":    train_data["pred_traj"][idx_val],
            "pred_traj_rel":train_data["pred_traj_rel"][idx_val],
        }
        if experiment_parameters.add_social:
            validation_data["obs_flow"]=train_data["obs_flow"][idx_val]

        # Training dataset
        pickle_out = open('training_data_'+experiment_name+'.pickle',"wb")
        pickle.dump(training_data, pickle_out, protocol=2)
        pickle_out.close()

        # Test dataset
        pickle_out = open('test_data_'+experiment_name+'.pickle',"wb")
        pickle.dump(test_data, pickle_out, protocol=2)
        pickle_out.close()

        # Validation dataset
        pickle_out = open('validation_data_'+experiment_name+'.pickle',"wb")
        pickle.dump(validation_data, pickle_out, protocol=2)
        pickle_out.close()
    else:
        # Unpickle the ready-to-use datasets
        print("[INF] Unpickling...")
        pickle_in = open('training_data_'+experiment_name+'.pickle',"rb")
        training_data = pickle.load(pickle_in)
        pickle_in = open('test_data_'+experiment_name+'.pickle',"rb")
        test_data = pickle.load(pickle_in)
        pickle_in = open('validation_data_'+experiment_name+'.pickle',"rb")
        validation_data = pickle.load(pickle_in)

    print("[INF] Training data: "+ str(len(training_data[list(training_data.keys())[0]])))
    print("[INF] Test data: "+ str(len(test_data[list(test_data.keys())[0]])))
    print("[INF] Validation data: "+ str(len(validation_data[list(validation_data.keys())[0]])))
    test_bckgd = mpimg.imread(testing_data_paths[0]+'/reference.png')
    # Load the homography corresponding to this dataset
    homography_file = os.path.join(testing_data_paths[0]+'/H.txt')
    test_homography = np.genfromtxt(homography_file)
    return training_data,validation_data,test_data,test_bckgd,test_homography
