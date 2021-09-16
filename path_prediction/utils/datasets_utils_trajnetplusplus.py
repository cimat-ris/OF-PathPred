import os, pickle, logging
import numpy as np
from .process_file_trajnetplusplus import prepare_data_trajnetplusplus

def setup_trajnetplusplus_experiment(experiment_name,ds_path,train_ds_names,test_ds_names,experiment_parameters,use_pickled_data=False,pickle_dir='pickle/',validation_proportion=0.1):
    # Dataset to be tested
    logging.info('Testing/validation dataset: {}'.format(test_ds_names))
    logging.info('Training datasets: {}'.format(train_ds_names))
    if use_pickled_data==False:
        # Process data specified by the path to get the trajectories with
        logging.info('Extracting data from the datasets '+ds_path)
        # Note that for the training data, we do not keep the neighbots information (too heavy!)
        train_data, train_primary_path = prepare_data_trajnetplusplus(ds_path+"/train/",train_ds_names,experiment_parameters,keep_neighbors=False)
        test_data, test_primary_path   = prepare_data_trajnetplusplus(ds_path+"/test_private/",test_ds_names, experiment_parameters,keep_neighbors=False)
        val_data, val_primary_path     = prepare_data_trajnetplusplus(ds_path+"/val/",test_ds_names, experiment_parameters,keep_neighbors=False)

        # Training dataset
        pickle_out = open(pickle_dir+'/training_data_'+experiment_name+'.pickle',"wb")
        #pickle.dump(train_data, pickle_out, protocol=2)
        pickle_out.close()

        # Test dataset
        pickle_out = open(pickle_dir+'/test_data_'+experiment_name+'.pickle',"wb")
        #pickle.dump(test_data, pickle_out, protocol=2)
        pickle_out.close()

        # Validation dataset
        pickle_out = open(pickle_dir+'/validation_data_'+experiment_name+'.pickle',"wb")
        #pickle.dump(val_data, pickle_out, protocol=2)
        pickle_out.close()
    else:
        # Unpickle the ready-to-use datasets
        logging.info("Unpickling...")
        pickle_in = open(pickle_dir+'/training_data_'+experiment_name+'.pickle',"rb")
        train_data = pickle.load(pickle_in)
        pickle_in = open(pickle_dir+'/test_data_'+experiment_name+'.pickle',"rb")
        test_data = pickle.load(pickle_in)
        pickle_in = open(pickle_dir+'/validation_data_'+experiment_name+'.pickle',"rb")
        val_data = pickle.load(pickle_in)

    logging.info("Training data: "+ str(len(train_data[list(train_data.keys())[0]])))
    logging.info("Test data: "+ str(len(test_data[list(test_data.keys())[0]])))
    logging.info("Validation data: "+ str(len(val_data[list(val_data.keys())[0]])))
    return train_data,val_data,test_data, train_primary_path, val_primary_path,test_primary_path
