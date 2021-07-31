import os, pickle, logging
import numpy as np
from path_prediction.process_file import prepare_data
from path_prediction.process_file import prepare_data_trajnetplusplus

def setup_loo_experiment(experiment_name,ds_path,ds_names,leave_id,experiment_parameters,use_pickled_data=False,pickle_dir='pickle/',validation_proportion=0.1):
    # Dataset to be tested
    testing_datasets_names  = [ds_names[leave_id]]
    training_datasets_names = ds_names[:leave_id]+ds_names[leave_id+1:]
    logging.info('Testing/validation dataset: {}'.format(testing_datasets_names))
    logging.info('Training datasets: {}'.format(training_datasets_names))
    if not use_pickled_data:
        # Process data specified by the path to get the trajectories with
        logging.info('Extracting data from the datasets')
        test_data  = prepare_data(ds_path, testing_datasets_names, experiment_parameters)
        train_data = prepare_data(ds_path, training_datasets_names, experiment_parameters)

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
            "obs_traj":        train_data["obs_traj"][idx_train],
            "obs_traj_rel":    train_data["obs_traj_rel"][idx_train],
            "obs_traj_theta":  train_data["obs_traj_theta"][idx_train],
            "pred_traj":       train_data["pred_traj"][idx_train],
            "pred_traj_rel":   train_data["pred_traj_rel"][idx_train],
            "frames_ids":      train_data["frames_ids"][idx_train],
            "obs_optical_flow":train_data["obs_optical_flow"][idx_train]
        }
        # Test set
        testing_data = {
            "obs_traj":      test_data["obs_traj"][:],
            "obs_traj_rel":  test_data["obs_traj_rel"][:],
            "obs_traj_theta":test_data["obs_traj_theta"][:],
            "pred_traj":     test_data["pred_traj"][:],
            "pred_traj_rel": test_data["pred_traj_rel"][:],
            "frames_ids":    test_data["frames_ids"][:],
            "obs_optical_flow": test_data["obs_optical_flow"][:]
        }
        # Validation set
        validation_data ={
            "obs_traj":      train_data["obs_traj"][idx_val],
            "obs_traj_rel":  train_data["obs_traj_rel"][idx_val],
            "obs_traj_theta":train_data["obs_traj_theta"][idx_val],
            "pred_traj":     train_data["pred_traj"][idx_val],
            "pred_traj_rel": train_data["pred_traj_rel"][idx_val],
            "frames_ids":    train_data["frames_ids"][idx_val],
            "obs_optical_flow": train_data["obs_optical_flow"][idx_val]
        }
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
        logging.info("Unpickling...")
        pickle_in = open(pickle_dir+'/training_data_'+experiment_name+'.pickle',"rb")
        training_data = pickle.load(pickle_in)
        pickle_in = open(pickle_dir+'/test_data_'+experiment_name+'.pickle',"rb")
        test_data = pickle.load(pickle_in)
        pickle_in = open(pickle_dir+'/validation_data_'+experiment_name+'.pickle',"rb")
        validation_data = pickle.load(pickle_in)

    logging.info("Training data: "+ str(len(training_data[list(training_data.keys())[0]])))
    logging.info("Test data: "+ str(len(test_data[list(test_data.keys())[0]])))
    logging.info("Validation data: "+ str(len(validation_data[list(validation_data.keys())[0]])))

    # Load the homography corresponding to this dataset
    homography_file = os.path.join(ds_path+testing_datasets_names[0]+'/H.txt')
    test_homography = np.genfromtxt(homography_file)
    return training_data,validation_data,test_data,test_homography


def setup_trajnetplusplus_experiment(experiment_name,ds_path,train_ds_names,test_ds_names,experiment_parameters,use_pickled_data=False,pickle_dir='pickle/',validation_proportion=0.1):
    # Dataset to be tested
    logging.info('Testing/validation dataset: {}'.format(test_ds_names))
    logging.info('Training datasets: {}'.format(train_ds_names))
    if use_pickled_data==False:
        # Process data specified by the path to get the trajectories with
        logging.info('Extracting data from the datasets')
        # Note that for the training data, we do not keep the neighbots information (too heavy!)
        train_data = prepare_data_trajnetplusplus(ds_path+"/train/train/real_data/",train_ds_names,experiment_parameters,keep_neighbors=False)
        test_data  = prepare_data_trajnetplusplus(ds_path+"/test/test/real_data/",test_ds_names, experiment_parameters)

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
            "pred_traj_rel": train_data["pred_traj_rel"][idx_train]
        }
        training_data["obs_optical_flow"]=train_data["obs_optical_flow"][idx_train]
        # Validation set
        validation_data ={
            "obs_traj":      train_data["obs_traj"][idx_val],
            "obs_traj_rel":  train_data["obs_traj_rel"][idx_val],
            "obs_traj_theta":train_data["obs_traj_theta"][idx_val],
            "pred_traj":     train_data["pred_traj"][idx_val],
            "pred_traj_rel": train_data["pred_traj_rel"][idx_val]
        }
        validation_data["obs_optical_flow"]=train_data["obs_optical_flow"][idx_val]
        # Test set
        testing_data = {
            "obs_traj":      test_data["obs_traj"][:],
            "obs_traj_rel":  test_data["obs_traj_rel"][:],
            "obs_traj_theta":test_data["obs_traj_theta"][:],
            "pred_traj":     test_data["pred_traj"][:],
            "pred_traj_rel": test_data["pred_traj_rel"][:]
        }
        testing_data["obs_optical_flow"]=test_data["obs_optical_flow"][:]
        testing_data["obs_neighbors"]=test_data["obs_neighbors"][:]

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
        logging.info("Unpickling...")
        pickle_in = open(pickle_dir+'/training_data_'+experiment_name+'.pickle',"rb")
        training_data = pickle.load(pickle_in)
        pickle_in = open(pickle_dir+'/test_data_'+experiment_name+'.pickle',"rb")
        test_data = pickle.load(pickle_in)
        pickle_in = open(pickle_dir+'/validation_data_'+experiment_name+'.pickle',"rb")
        validation_data = pickle.load(pickle_in)

    logging.info("Training data: "+ str(len(training_data[list(training_data.keys())[0]])))
    logging.info("Test data: "+ str(len(test_data[list(test_data.keys())[0]])))
    logging.info("Validation data: "+ str(len(validation_data[list(validation_data.keys())[0]])))
    return training_data,validation_data,test_data
