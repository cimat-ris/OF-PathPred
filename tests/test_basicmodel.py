# -*- coding: utf-8 -*-
# Imports
import sys, os, argparse, logging, random, logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import math,numpy as np
import tensorflow as tf
# Important imports
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from path_prediction import models, utils
from path_prediction.models.model_deterministic_rnn import PredictorDetRNN

def main():
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='datasets/',
                        help='glob expression for data files')
    parser.add_argument('--obstacles', dest='obstacles', action='store_true',help='includes the obstacles in the optical flow')
    parser.set_defaults(obstacles=False)
    parser.add_argument('--log_level',type=int, default=20,help='Log level (default: 20)')
    parser.add_argument('--log_file',default='',help='Log file (default: standard output)')
    parser.add_argument('--pickle', dest='pickle', action='store_true',help='uses previously pickled data')
    parser.set_defaults(pickle=False)
    parser.add_argument('--cpu', dest='cpu', action='store_true',help='Use CPU')
    parser.set_defaults(cpu=False)
    parser.add_argument('--dataset_id', '--id',
                    type=int, default=0,help='dataset id (default: 0)')
    parser.add_argument('--noretrain', dest='noretrain', action='store_true',help='When set, does not retrain the model, and only restores the last checkpoint')
    parser.set_defaults(noretrain=False)
    parser.add_argument('--epochs', '--e',
                    type=int, default=20,help='Number of epochs (default: 20)')
    parser.add_argument('--rnn', default='lstm', choices=['gru', 'lstm'],
                    help='recurrent networks to be used (default: "lstm")')
    parser.add_argument('--coords_mode', default='rel_rot', choices=['rel', 'rel_rot'],
                    help='coordinates to be used as input (default: "rel_rot")')
    args = parser.parse_args()

    if args.log_file=='':
        logging.basicConfig(format='%(levelname)s: %(message)s',level=args.log_level)
    else:
        logging.basicConfig(filename=args.log_file,format='%(levelname)s: %(message)s',level=args.log_level)

    logging.info('Tensorflow version: {}'.format(tf.__version__))
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices)>0 and args.cpu==False:
        logging.info('Using GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        tf.config.set_visible_devices([], 'GPU')
        logging.info('Using CPU')
    # Load the default parameters
    experiment_parameters = utils.training_utils.Experiment_Parameters(obstacles=False)
    dataset_dir   = args.path
    dataset_names = ['eth-hotel','eth-univ','ucy-zara01','ucy-zara02','ucy-univ']
    dataset_flips = [True,False,False,False,False]
    # Load the dataset and perform the split
    idTest = args.dataset_id
    training_data,validation_data,test_data,test_homography = utils.datasets_utils.setup_loo_experiment('ETH_UCY',dataset_dir,dataset_names,idTest,experiment_parameters,use_pickled_data=args.pickle)

    #############################################################
    # Model parameters
    model_parameters = PredictorDetRNN.parameters(rnn_type=args.rnn)
    model_parameters.num_epochs     = args.epochs
    model_parameters.emb_size       = 64
    model_parameters.enc_hidden_size= 256
    model_parameters.dec_hidden_size= model_parameters.enc_hidden_size
    model_parameters.dropout        = 0.3
    model_parameters.coords_mode    = args.coords_mode
    logging.info('Using coordinate mode '+args.coords_mode)
    # Get the necessary data
    train_data = tf.data.Dataset.from_tensor_slices(training_data)
    val_data   = tf.data.Dataset.from_tensor_slices(validation_data)
    test_data  = tf.data.Dataset.from_tensor_slices(test_data)

    # Form batches
    batched_train_data = train_data.batch(model_parameters.batch_size)
    batched_val_data   = val_data.batch(model_parameters.batch_size)
    batched_test_data  = test_data.batch(model_parameters.batch_size)

    # Model
    model     = PredictorDetRNN(model_parameters)
    optimizer = optimizers.Adam(learning_rate=1.5e-6)

    # Checkpoints
    checkpoint_dir   = './training_checkpoints/basicmodel'
    checkpoint_prefix= os.path.join(checkpoint_dir, "ckpt")
    checkpoint       = tf.train.Checkpoint(optimizer=optimizer,
                                        weights=model)
    # Training the Model
    train_loss_results = []
    val_loss_results   = []
    val_metrics_results= {'mADE': [], 'mFDE': []}

    if args.noretrain==False:
        logging.info("Training the model")
        for epoch in range(model_parameters.num_epochs ):
            # Training
            logging.info("----- ")
            logging.info("Epoch: {}".format(epoch+1))
            total_error = 0
            total_cases = 0
            num_batches_per_epoch= batched_train_data.cardinality().numpy()
            for batch_idx, dicto in enumerate(batched_train_data):
                condition  = dicto['obs_traj_'+args.coords_mode]
                target     = dicto['pred_traj_'+args.coords_mode]
                with tf.GradientTape() as tape:
                    losses      = model(condition, target, training=True)
                    total_error+= losses
                    total_cases+= num_batches_per_epoch
                gradients = tape.gradient(losses, model.trainable_weights)
                optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            logging.info("Training loss: {:.6f}".format(total_error.numpy()/total_cases))
            train_loss_results.append(total_error/total_cases)
            # Validation
            total_error = 0
            total_cases = 0
            for batch_idx, dicto in enumerate(batched_val_data):
                condition  = dicto['obs_traj_'+args.coords_mode]
                target     = dicto['pred_traj_'+args.coords_mode]
                losses = model(condition, target)
                total_error += losses
                total_cases += num_batches_per_epoch
            logging.info("Validation loss: {:.6f}".format(total_error.numpy()/total_cases))
            val_loss_results.append(total_error/total_cases)
            # Evaluate ADE, FDE metrics on validation data
            val_quantitative_metrics = utils.testing_utils.evaluation_minadefde(model,batched_val_data,model_parameters)
            val_metrics_results['mADE'].append(val_quantitative_metrics['mADE'])
            val_metrics_results['mFDE'].append(val_quantitative_metrics['mFDE'])
            # Saving (checkpoint) the model every 2 epochs
            if (epoch + 1) % 5 == 0:
                logging.info('Saving current model')
                checkpoint.save(file_prefix = checkpoint_prefix)
            logging.info('Epoch {}. Validation mADE {:.4f}'.format(epoch + 1, val_quantitative_metrics['mADE']))
            logging.info('Epoch {}. Validation mFDE {:.4f}'.format(epoch + 1, val_quantitative_metrics['mFDE']))

        plot_training    = True
        if plot_training==True:
            utils.plot_utils.plot_training_results(train_loss_results,val_loss_results,val_metrics_results)


    # Testing
    # Restoring the latest checkpoint in checkpoint_dir
    logging.info("Restoring best model")
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    # Quantitative testing: ADE/FDE
    quantitative = True
    if quantitative==True:
        logging.info("Quantitative testing")
        results = utils.testing_utils.evaluation_minadefde(model,batched_test_data,model_parameters)
        utils.testing_utils.plot_comparisons_minadefde(results,dataset_names[idTest])
        logging.info(results)
    # Qualitative testing
    qualitative = True
    if qualitative==True:
        logging.info("Qualitative testing")
        for i in range(5):
            batch, test_bckgd = utils.testing_utils.get_testing_batch(test_data,dataset_dir+dataset_names[idTest])
            utils.testing_utils.evaluation_qualitative(model,batch,model_parameters,background=test_bckgd,homography=test_homography, flip=dataset_flips[args.dataset_id],n_peds_max=1)
        logging.info("Worst cases")
        utils.testing_utils.evaluation_worstcases(model,batched_test_data,model_parameters,background=test_bckgd,homography=test_homography, flip=dataset_flips[args.dataset_id])


if __name__ == '__main__':
    main()
