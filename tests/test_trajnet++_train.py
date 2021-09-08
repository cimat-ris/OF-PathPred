import sys, os, argparse, logging,random, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import tensorflow as tf
from datetime import datetime
random.seed(datetime.now())
import matplotlib.pyplot as plt

# Important imports
from path_prediction import models, utils
import pickle

sys.path.append("../trajnetplusplusbaselines")
from trajnetplusplusbaselines.evaluator.design_pd import Table

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs_length', default=8, type=int,help='observation length')
    parser.add_argument('--pred_length', default=12, type=int,help='prediction length')
    parser.add_argument('--path', default='datasets/trajnetplusplus',help='glob expression for data files')
    #parser.add_argument('--path', default='../trajnetplusplusdataset/output/',help='glob expression for data files')
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
    parser.add_argument('--coords_mode', default='rel', choices=['rel', 'rel_rot'],
                    help='coordinates to be used as input (default: "rel")')

    # arg for evaluation trajnetplusplus
    parser.add_argument('--output', nargs='+',
                    help='relative path to saved model')
    parser.add_argument('--sf', action='store_true',
                        help='consider socialforce in evaluation')
    parser.add_argument('--orca', action='store_true',
                        help='consider orca in evaluation')
    parser.add_argument('--kf', action='store_true',
                        help='consider kalman in evaluation')
    parser.add_argument('--cv', action='store_true',
                        help='consider constant velocity in evaluation')
    parser.add_argument('--normalize_scene', action='store_true',
                        help='augment scenes')
    parser.add_argument('--modes', default=1, type=int,
                        help='number of modes to predict')
    parser.add_argument('--labels', required=False, nargs='+',
                        help='labels of models')
    parser.add_argument('--disable-collision', action='store_true',
                        help='disable collision metrics')

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

    ## List of .json file inside the args.path (waiting to be predicted by the testing model)
    datasets = sorted([f.split('.')[-2] for f in os.listdir(args.path+'/train') if not f.startswith('.') and f.endswith('.ndjson')])
    logging.info("Found the following datasets: {}".format(datasets))
    train_dataset_names = datasets
    test_dataset_names  = datasets
    # Load the default parameters
    experiment_parameters = utils.training_utils.Experiment_Parameters(obstacles=False)
    experiment_parameters.obs_len  = args.obs_length
    experiment_parameters.pred_len = args.pred_length
    # Load the datasets
    training_data,validation_data,testing_data, train_primary_path,val_primary_path, test_primary_path = utils.datasets_utils.setup_trajnetplusplus_experiment('TRAJNETPLUSPLUS',args.path,train_dataset_names,test_dataset_names,experiment_parameters,use_pickled_data=args.pickle)
    logging.info("Total number of training trajectories: {}".format(training_data["obs_traj"].shape[0]))

    #############################################################
    # Model parameters
    model_parameters = models.model_multimodal_of.PredictorMultOf.Parameters(rnn_type=args.rnn)
    model_parameters.num_epochs     = args.epochs
    # 9 samples generated
    model_parameters.add_social     = args.social
    model_parameters.coords_mode    = args.coords_mode
    model_parameters.output_var_dirs= 0
    model_parameters.initial_lr     = 0.03
    model_parameters.pred_length    = 12
    model_parameters.obs_length     = 9
    model_parameters.use_validation = True

    # Get the necessary data as tensorflow datasets
    train_data = tf.data.Dataset.from_tensor_slices(training_data)
    val_data   = tf.data.Dataset.from_tensor_slices(validation_data)
    test_data  = tf.data.Dataset.from_tensor_slices(testing_data)

    # Form batches
    batched_train_data = train_data.batch(model_parameters.batch_size)
    batched_val_data   = val_data.batch(model_parameters.batch_size)
    batched_test_data  = test_data.batch(model_parameters.batch_size)

    # Model
    tj_enc_dec = models.model_multimodal_of.PredictorMultOf(model_parameters)

    # Checkpoints
    checkpoint_dir   = 'training_checkpoints/ofmodel-trajnet++'
    checkpoint_prefix= os.path.join(checkpoint_dir, "ckpt")
    checkpoint       = tf.train.Checkpoint(encoder=tj_enc_dec.enc,
                                            enctodec = tj_enc_dec.enctodec,
                                            decoder=tj_enc_dec.dec)

    # Training
    if args.noretrain==False:
        logging.info("Training the model")
        train_loss_results,val_loss_results,val_metrics_results,__ = utils.training_utils.training_trajnetplusplus_loop(tj_enc_dec,batched_train_data,batched_val_data,model_parameters,checkpoint,checkpoint_prefix)
        utils.plot_utils.plot_training_results(train_loss_results,val_loss_results,val_metrics_results)

    # Testing
    # Restoring the latest checkpoint in checkpoint_dir
    logging.info("Restoring last model")
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # Quantitative testing: ADE/FDE
    quantitative = True
    add_info_other_models = True

    if quantitative==True:
        logging.info("Quantitative testing")

        # Initiate Result Table
        table = Table()

        # Calulate trajnetplusplus metrics
        utils.testing_utils.evaluation_trajnetplusplus_minadefde(tj_enc_dec, batched_test_data, test_primary_path, model_parameters, table=table)

        # For add result of model orca, kf, cv, sf
        # if add_info_other_models:
        #    args.obs_length = 9
        #    utils.testing_utils.evaluation_trajnetplusplus_other_models(args,table)

        ## Save table in a image
        table.print_table()
        print("results saved in Results.png")


if __name__ == '__main__':
    main()
