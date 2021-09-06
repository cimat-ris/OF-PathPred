# Imports
import sys, os, argparse, logging,random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import math,numpy as np
import tensorflow as tf
# Important imports
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from path_prediction import models, utils
from path_prediction.models.model_multimodal_attention import PredictorMultAtt
from path_prediction.models.model_multimodal_of import PredictorMultOf

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
    parser.add_argument('--model', '--m',choices=['attention', 'of'],default='attention',
                    help='model (default: 0)')
    parser.add_argument('--social', dest='social', action='store_true',help='Models social interactions')
    parser.set_defaults(social=False)
    parser.add_argument('--log_polar', dest='log_polar', action='store_true',help='Use log polar mapping for synthesized optical flow')
    parser.set_defaults(log_polar=False)
    parser.add_argument('--qualitative', dest='qualitative', action='store_true',help='Performs a qualitative evaluation')
    parser.set_defaults(qualitative=False)
    parser.add_argument('--quantitative', dest='quantitative', action='store_true',help='Performs a quantitative evaluation')
    parser.set_defaults(quantitative=False)
    parser.add_argument('--noretrain', dest='noretrain', action='store_true',help='When set, does not retrain the model, and only restores the last checkpoint')
    parser.set_defaults(noretrain=False)
    parser.add_argument('--epochs', '--e',
                    type=int, default=30,help='Number of epochs (default: 25)')
    parser.add_argument('--rnn', default='lstm', choices=['gru', 'lstm'],
                    help='recurrent networks to be used (default: "lstm")')
    parser.add_argument('--coords_mode', default='rel_rot', choices=['rel', 'rel_rot'],
                    help='coordinates to be used as input (default: "rel_rot")')
    parser.add_argument('--pickle', dest='pickle', action='store_true',help='uses previously pickled data')
    parser.set_defaults(pickle=False)
    parser.add_argument('--cpu', dest='cpu', action='store_true',help='Use CPU')
    parser.set_defaults(cpu=False)
    args = parser.parse_args()
    if args.log_file=='':
        logging.basicConfig(format='%(levelname)s: %(message)s',level=args.log_level)
    else:
        logging.basicConfig(filename=args.log_file,format='%(levelname)s: %(message)s',level=args.log_level)

    # Info about TF and GPU
    logging.info('Tensorflow version: {}'.format(tf.__version__))
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices)>0 and args.cpu==False:
        logging.info('Using GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        tf.config.set_visible_devices([], 'GPU')
        logging.info('Using CPU')


    # Load the default parameters
    experiment_parameters = utils.training_utils.Experiment_Parameters(obstacles=args.obstacles)
    experiment_parameters.log_polar_mapping = args.log_polar
    dataset_dir   = args.path
    dataset_names = ['eth-hotel','eth-univ','ucy-zara01','ucy-zara02','ucy-univ']
    dataset_flips = [True,False,False,False,False]

    # Load the dataset and perform the split
    idTest = args.dataset_id
    training_data,validation_data,test_data,test_homography = utils.datasets_utils.setup_loo_experiment('ETH_UCY',dataset_dir,dataset_names,idTest,experiment_parameters,use_pickled_data=args.pickle)

    #############################################################
    # Model parameters
    if args.model=="of":
        model_parameters = PredictorMultOf.Parameters(rnn_type=args.rnn)
    else:
        if args.model=="attention":
            model_parameters = PredictorMultAtt.Parameters(add_social=args.social,rnn_type=args.rnn)
        else:
            logging.error("No such model")
    model_parameters.num_epochs    = args.epochs
    model_parameters.add_social    = args.social
    # Number of samples generated
    model_parameters.output_var_dirs= 3
    model_parameters.coords_mode    = args.coords_mode
    logging.info('Using coordinate mode '+args.coords_mode)
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
    batched_train_data = train_data.shuffle(buffer_size=20000, reshuffle_each_iteration=True).batch(model_parameters.batch_size)
    batched_val_data   = val_data.batch(model_parameters.batch_size)
    batched_test_data  = test_data.batch(model_parameters.batch_size)

    # Model
    if args.model=="of":
        tj_enc_dec = PredictorMultOf(model_parameters)
    else:
        tj_enc_dec = PredictorMultAtt(model_parameters)
    tj_enc_dec
    # Checkpoints
    checkpoint_dir   = './training_checkpoints/ofmodel-'+args.model
    checkpoint_prefix= os.path.join(checkpoint_dir, "ckpt")
    checkpoint       = tf.train.Checkpoint(optimizer=tj_enc_dec.optimizer,
                                        encoder=tj_enc_dec.enc,
                                        enctodec = tj_enc_dec.enctodec,
                                        decoder=tj_enc_dec.dec)

    # Training
    if args.noretrain==False:
        logging.info("Training the model")
        train_loss_results,val_loss_results,val_metrics_results,__ = utils.training_utils.training_loop(tj_enc_dec,batched_train_data,batched_val_data,model_parameters,checkpoint,checkpoint_prefix)
        utils.plot_utils.plot_training_results(train_loss_results,val_loss_results,val_metrics_results)

    # Testing
    # Restoring the latest checkpoint in checkpoint_dir
    logging.info("Restoring last model")
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    # Quantitative testing: ADE/FDE
    if args.quantitative==True:
        logging.info("Quantitative testing")
        results = utils.testing_utils.evaluation_minadefde(tj_enc_dec,batched_test_data,model_parameters)
        utils.testing_utils.plot_comparisons_minadefde(results,dataset_names[idTest])
        logging.info(results)

    # Qualitative testing
    if args.qualitative==True:
        logging.info("Qualitative testing")
        for i in range(5):
            batch, test_bckgd = utils.testing_utils.get_testing_batch(test_data,dataset_dir+dataset_names[idTest])
            utils.testing_utils.evaluation_qualitative(tj_enc_dec,batch,model_parameters,background=test_bckgd,homography=test_homography, flip=dataset_flips[args.dataset_id],n_peds_max=1)
        logging.info("Worst cases")
        utils.testing_utils.evaluation_worstcases(tj_enc_dec,batched_test_data,model_parameters,background=None,homography=None, flip=dataset_flips[args.dataset_id])

if __name__ == '__main__':
    main()
