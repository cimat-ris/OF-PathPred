# -*- coding: utf-8 -*-
# Imports
import argparse
import sys,os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math,numpy as np
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
print('[INF] Tensorflow version: ',tf.__version__)
tf.test.gpu_device_name()
# Important imports
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from path_prediction.datasets_utils import setup_loo_experiment
from path_prediction.model import BasicRNNModel, BasicRNNModelParameters
from path_prediction.plot_utils import plot_training_data,plot_training_results
import path_prediction.batches_data
from path_prediction.testing_utils import evaluation_minadefde,evaluation_qualitative,evaluation_attention,plot_comparisons_minadefde, get_testing_batch
from path_prediction.training_utils import Experiment_Parameters
from path_prediction.testing_utils import evaluation_minadefde


def main():
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='datasets/',
                        help='glob expression for data files')
    parser.add_argument('--obstacles', dest='obstacles', action='store_true',help='includes the obstacles in the optical flow')
    parser.set_defaults(obstacles=False)
    parser.add_argument('--dataset_id', '--id',
                    type=int, default=0,help='dataset id (default: 0)')
    parser.add_argument('--epochs', '--e',
                    type=int, default=35,help='Number of epochs (default: 35)')
    parser.add_argument('--rnn', default='lstm', choices=['gru', 'lstm'],
                    help='recurrent networks to be used (default: "lstm")')
    args = parser.parse_args()

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices)>0:
        print('[INF] Using GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("[INF] Using CPU")

    # Load the default parameters
    experiment_parameters = Experiment_Parameters(add_social=False,add_kp=False,obstacles=args.obstacles)

    dataset_dir   = args.path
    dataset_names = ['eth-hotel','eth-univ','ucy-zara01','ucy-zara02','ucy-univ']

    # Load the dataset and perform the split
    idTest = args.dataset_id
    training_data,validation_data,test_data,test_homography = setup_loo_experiment('ETH_UCY',dataset_dir,dataset_names,idTest,experiment_parameters,use_pickled_data=True)

    # Plot ramdomly a subset of the training data (spatial data only)
    show_training_samples = False
    if show_training_samples:
        plot_training_data(training_data,experiment_parameters)

    #############################################################
    # Model parameters
    model_parameters = BasicRNNModelParameters()
    model_parameters.num_epochs     = args.epochs
    model_parameters.initial_lr     = 0.03

    # Get the necessary data
    train_data = tf.data.Dataset.from_tensor_slices(training_data)
    val_data   = tf.data.Dataset.from_tensor_slices(validation_data)
    test_data  = tf.data.Dataset.from_tensor_slices(test_data)

    # Form batches
    batched_train_data = train_data.batch(model_parameters.batch_size)
    batched_val_data   = val_data.batch(model_parameters.batch_size)
    batched_test_data  = test_data.batch(model_parameters.batch_size)

    # Model
    model     = BasicRNNModel(model_parameters)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

    # Training the Model
    train_loss_results = []
    val_loss_results   = []
    val_metrics_results= {'mADE': [], 'mFDE': []}

    print("[INF] Training the model")
    for epoch in range(model_parameters.num_epochs ):
        # Training
        print("----- ")
        print("Epoch: ", epoch)
        error = 0
        total = 0
        for batch_idx, dicto in enumerate(batched_train_data):
            data   = dicto['obs_traj_rel']
            target = dicto['pred_traj_rel']
            with tf.GradientTape() as tape:
                losses  = model(data, target, training=True)
                avg_loss= tf.reduce_sum(losses)/losses.shape[0]
                error  += tf.reduce_sum(losses)
                total  += losses.shape[0]
            gradients = tape.gradient(avg_loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        print("Training loss: ", error.numpy()/total)
        train_loss_results.append(error/total)
        # Validation
        error = 0
        total = 0
        for batch_idx, dicto in enumerate(batched_val_data):
            data_val   = dicto['obs_traj_rel']
            target_val = dicto['pred_traj_rel']
            losses = model(data_val, target_val)
            error  += tf.reduce_sum(losses)
            total += losses.shape[0]
        print("Validation loss: ", error.numpy()/total)
        val_loss_results.append(error/total)
        # Evaluate ADE, FDE metrics on validation data
        val_quantitative_metrics = evaluation_minadefde(model,batched_val_data,model_parameters)
        val_metrics_results['mADE'].append(val_quantitative_metrics['mADE'])
        val_metrics_results['mFDE'].append(val_quantitative_metrics['mFDE'])

    # Checkpoints
    checkpoint_dir   = './training_checkpoints/basicmodel'
    checkpoint_prefix= os.path.join(checkpoint_dir, "ckpt")
    checkpoint       = tf.train.Checkpoint(optimizer=optimizer,
                                        weights=model)

    # Training
    plot_training    = True
    if plot_training==True:
        plot_training_results(train_loss_results,val_loss_results,val_metrics_results)

    # Testing
    # Restoring the latest checkpoint in checkpoint_dir
    print("[INF] Restoring last model")
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # Quantitative testing: ADE/FDE
    quantitative = True
    if quantitative==True:
        print("[INF] Quantitative testing")
        results = evaluation_minadefde(model,batched_test_data,model_parameters)
        plot_comparisons_minadefde(results,dataset_names[idTest])
        print(results)

    # Qualitative testing
    qualitative = True
    if qualitative==True:
        print("[INF] Qualitative testing")
        for i in range(5):
            batch, test_bckgd = get_testing_batch(test_data,dataset_dir+dataset_names[idTest])
            evaluation_qualitative(model,batch,model_parameters,background=test_bckgd,homography=test_homography, flip=False,n_peds_max=1,display_mode=None)



if __name__ == '__main__':
    main()