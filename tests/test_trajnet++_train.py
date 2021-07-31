import sys, os, argparse, logging,random, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import tensorflow as tf
from datetime import datetime
random.seed(datetime.now())
import matplotlib.pyplot as plt

# Important imports
import path_prediction.batches_data
from path_prediction.training_utils import Experiment_Parameters
from path_prediction.interaction_optical_flow import OpticalFlowSimulator
from path_prediction.process_file import prepare_data_trajnetplusplus
from path_prediction.datasets_utils import setup_trajnetplusplus_experiment
from path_prediction.datasets_utils import setup_loo_experiment
from path_prediction.model import TrajectoryEncoderDecoder, ModelParameters
from path_prediction.plot_utils import plot_training_data,plot_training_results
from path_prediction.training_utils import training_loop
import path_prediction.batches_data
from path_prediction.testing_utils import evaluation_minadefde,evaluation_qualitative,evaluation_attention,plot_comparisons_minadefde, get_testing_batch

import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs_length', default=8, type=int,help='observation length')
    parser.add_argument('--pred_length', default=12, type=int,help='prediction length')
    parser.add_argument('--path', default='datasets/trajnetplusplus',help='glob expression for data files')
    parser.add_argument('--log_level',type=int, default=20,help='Log level (default: 20)')
    parser.add_argument('--log_file',default='',help='Log file (default: standard output)')
    parser.add_argument('--pickle', dest='pickle', action='store_true',help='uses previously pickled data')
    parser.set_defaults(pickle=False)
    parser.add_argument('--social', dest='social', action='store_true',help='Models social interactions')
    parser.set_defaults(social=False)
    parser.add_argument('--noretrain', dest='noretrain', action='store_true',help='When set, does not retrain the model, and only restores the last checkpoint')
    parser.set_defaults(noretrain=False)
    parser.add_argument('--epochs', '--e',
                    type=int, default=35,help='Number of epochs (default: 35)')
    parser.add_argument('--rnn', default='lstm', choices=['gru', 'lstm'],
                    help='recurrent networks to be used (default: "lstm")')
    args     = parser.parse_args()
    if args.log_file=='':
        logging.basicConfig(format='%(levelname)s: %(message)s',level=args.log_level)
    else:
        logging.basicConfig(filename=args.log_file,format='%(levelname)s: %(message)s',level=args.log_level)
    # Info about TF and GPU
    logging.info('Tensorflow version: {}'.format(tf.__version__))
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices)>0:
        logging.info('Using GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        logging.info('Using CPU')


    train_dataset_names = ["biwi_hotel","crowds_students001","crowds_students003","crowds_zara01","crowds_zara03","lcas","wildtrack","cff_06","cff_07","cff_08"]
    #train_dataset_names = ["biwi_hotel","crowds_students001","crowds_students003"]
    test_dataset_names = ["biwi_eth"]

    # Load the default parameters
    experiment_parameters = Experiment_Parameters(add_kp=False)
    experiment_parameters.obs_len  = args.obs_length
    experiment_parameters.pred_len = args.pred_length
    # Load the datasets
    training_data,validation_data,testing_data = setup_trajnetplusplus_experiment('TRAJNETPLUSPLUS',args.path,train_dataset_names,test_dataset_names,experiment_parameters,use_pickled_data=args.pickle)
    logging.info("Total number of training trajectories: {}".format(training_data["obs_traj"].shape[0]))

    #############################################################
    # Model parameters
    model_parameters = ModelParameters(add_attention=True,add_kp=experiment_parameters.add_kp,add_social=args.social,rnn_type=args.rnn)
    model_parameters.num_epochs     = args.epochs
    # 9 samples generated
    model_parameters.output_var_dirs= 1
    model_parameters.is_mc_dropout  = False
    model_parameters.initial_lr     = 0.03
    model_parameters.enc_hidden_size= 128  # Hidden size of the RNN encoder
    model_parameters.dec_hidden_size= model_parameters.enc_hidden_size # Hidden size of the RNN decoder
    model_parameters.emb_size       = 256  # Embedding size
    model_parameters.use_validation = False
    # When running on CPU
    if len(physical_devices)==0:
        model_parameters.batch_size     = 64
        model_parameters.output_var_dirs= 1
        model_parameters.stack_rnn_size = 1

    # Get the necessary data as tensorflow datasets
    train_data = tf.data.Dataset.from_tensor_slices(training_data)
    val_data   = tf.data.Dataset.from_tensor_slices(validation_data)
    test_data  = tf.data.Dataset.from_tensor_slices(testing_data)

    # Form batches
    batched_train_data = train_data.batch(model_parameters.batch_size)
    batched_val_data   = val_data.batch(model_parameters.batch_size)
    batched_test_data  = test_data.batch(model_parameters.batch_size)

    # Model
    tj_enc_dec = TrajectoryEncoderDecoder(model_parameters)

    # Checkpoints
    checkpoint_dir   = './training_checkpoints/ofmodel-trajnet++'
    checkpoint_prefix= os.path.join(checkpoint_dir, "ckpt")
    checkpoint       = tf.train.Checkpoint(optimizer=tj_enc_dec.optimizer,
                                        encoder=tj_enc_dec.enc,
                                        decoder=tj_enc_dec.dec)

    # Training
    if args.noretrain==False:
        logging.info("Training the model")
        train_loss_results,val_loss_results,val_metrics_results,__ = training_loop(tj_enc_dec,batched_train_data,batched_val_data,model_parameters,checkpoint,checkpoint_prefix)
        plot_training_results(train_loss_results,val_loss_results,val_metrics_results)

    # Testing
    # Restoring the latest checkpoint in checkpoint_dir
    logging.info("Restoring last model")
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


if __name__ == '__main__':
    main()
