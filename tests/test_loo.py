# Imports
import sys, os, argparse, logging,random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math,numpy as np
import tensorflow as tf
# Important imports
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from path_prediction.datasets_utils import setup_loo_experiment
from path_prediction.model import TrajectoryEncoderDecoder, ModelParameters
from path_prediction.plot_utils import plot_training_data,plot_training_results
import path_prediction.batches_data
from path_prediction.testing_utils import evaluation_minadefde,evaluation_qualitative,evaluation_attention,plot_comparisons_minadefde, get_testing_batch
from path_prediction.training_utils import training_loop
from path_prediction.training_utils import Experiment_Parameters


def main():
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='datasets/',
                        help='glob expression for data files')
    parser.add_argument('--obstacles', dest='obstacles', action='store_true',help='includes the obstacles in the optical flow')
    parser.set_defaults(obstacles=False)
    parser.add_argument('--log_level',type=int, default=20,help='Log level (default: 20)')
    parser.add_argument('--log_file',default='',help='Log file (default: standard output)')
    parser.add_argument('--dataset_id', '--id',
                    type=int, default=0,help='dataset id (default: 0)')
    parser.add_argument('--social', dest='social', action='store_true',help='Models social interactions')
    parser.set_defaults(social=False)
    parser.add_argument('--noretrain', dest='noretrain', action='store_true',help='When set, does not retrain the model, and only restores the last checkpoint')
    parser.set_defaults(noretrain=False)
    parser.add_argument('--epochs', '--e',
                    type=int, default=35,help='Number of epochs (default: 35)')
    parser.add_argument('--rnn', default='lstm', choices=['gru', 'lstm'],
                    help='recurrent networks to be used (default: "lstm")')
    parser.add_argument('--pickle', dest='pickle', action='store_true',help='uses previously pickled data')
    parser.set_defaults(pickle=False)
    args = parser.parse_args()
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
        logging.info("Using CPU")

    # Load the default parameters
    experiment_parameters = Experiment_Parameters(add_kp=False,obstacles=args.obstacles)

    dataset_dir   = args.path
    dataset_names = ['eth-hotel','eth-univ','ucy-zara01','ucy-zara02','ucy-univ']

    # Load the dataset and perform the split
    idTest = args.dataset_id
    training_data,validation_data,test_data,test_homography = setup_loo_experiment('ETH_UCY',dataset_dir,dataset_names,idTest,experiment_parameters,use_pickled_data=args.pickle)

    # Plot ramdomly a subset of the training data (spatial data only)
    show_training_samples = False
    if show_training_samples:
        plot_training_data(training_data,experiment_parameters)

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

    # When running on CPU
    if len(physical_devices)==0:
        model_parameters.batch_size     = 64
        model_parameters.output_var_dirs= 1
        model_parameters.stack_rnn_size = 1

    # Get the necessary data
    train_data = tf.data.Dataset.from_tensor_slices(training_data)
    val_data   = tf.data.Dataset.from_tensor_slices(validation_data)
    test_data  = tf.data.Dataset.from_tensor_slices(test_data)

    # Form batches
    batched_train_data = train_data.batch(model_parameters.batch_size)
    batched_val_data   = val_data.batch(model_parameters.batch_size)
    batched_test_data  = test_data.batch(model_parameters.batch_size)

    # Model
    tj_enc_dec = TrajectoryEncoderDecoder(model_parameters)

    # Checkpoints
    checkpoint_dir   = './training_checkpoints/ofmodel'
    checkpoint_prefix= os.path.join(checkpoint_dir, "ckpt")
    checkpoint       = tf.train.Checkpoint(optimizer=tj_enc_dec.optimizer,
                                        encoder=tj_enc_dec.enc,
                                        decoder=tj_enc_dec.dec)

    # Training
    plot_training    = True
    if args.noretrain==False:
        logging.info("Training the model")
        train_loss_results,val_loss_results,val_metrics_results,__ = training_loop(tj_enc_dec,batched_train_data,batched_val_data,model_parameters,checkpoint,checkpoint_prefix)
        if plot_training==True:
            plot_training_results(train_loss_results,val_loss_results,val_metrics_results)

    # Testing
    # Restoring the latest checkpoint in checkpoint_dir
    logging.info("Restoring last model")
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # Quantitative testing: ADE/FDE
    quantitative = True
    if quantitative==True:
        logging.info("Quantitative testing")
        results = evaluation_minadefde(tj_enc_dec,batched_test_data,model_parameters)
        plot_comparisons_minadefde(results,dataset_names[idTest])
        logging.info(results)

    # Qualitative testing
    qualitative = True
    if qualitative==True:
        logging.info("Qualitative testing")
        for i in range(5):
            batch, test_bckgd = get_testing_batch(test_data,dataset_dir+dataset_names[idTest])
            #evaluation_qualitative(tj_enc_dec,batch,model_parameters,background=test_bckgd,homography=test_homography, flip=False,n_peds_max=1,display_mode=None)
            evaluation_attention(tj_enc_dec,batch,model_parameters,background=test_bckgd,homography=test_homography, flip=False,display_mode=None)



if __name__ == '__main__':
    main()
