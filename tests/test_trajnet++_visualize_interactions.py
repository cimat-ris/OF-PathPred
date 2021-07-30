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
from trajnetplusplustools.trajnetplusplustools import load_all
from trajnetplusplustools.trajnetplusplustools import show
from trajnetplusplustools.trajnetplusplustools.summarize import dataset_plots
from trajnetplusplustools.trajnetplusplustools.visualize_type import interaction_plots

import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs_len', default=8, type=int,help='observation length')
    parser.add_argument('--pred_len', default=12, type=int,help='prediction length')
    parser.add_argument('--path', default='datasets/trajnetplusplus',help='glob expression for data files')
    parser.add_argument('--log_level',type=int, default=20,help='Log level (default: 20)')
    parser.add_argument('--log_file',default='',help='Log file (default: standard output)')
    parser.add_argument('--n', type=int, default=5,
                        help='number of samples')
    parser.add_argument('--trajectory_type', type=int, default=3,
                        help='type of trajectory (2: Lin, 3: NonLin + Int, 4: NonLin + NonInt)')
    parser.add_argument('--interaction_type', type=int, default=2,
                        help='type of interaction (1: LF, 2: CA, 3:Grp, 4:Oth)')
    parser.add_argument('--pos_angle', type=int, default=0,
                        help='axis angle of position cone (in deg)')
    parser.add_argument('--vel_angle', type=int, default=0,
                        help='relative velocity centre (in deg)')
    parser.add_argument('--pos_range', type=int, default=15,
                        help='range of position cone (in deg)')
    parser.add_argument('--vel_range', type=int, default=20,
                        help='relative velocity span (in rsdeg)')
    parser.add_argument('--dist_thresh', type=int, default=5,
                        help='threshold of distance (in m)')
    parser.add_argument('--choice', default='bothpos',
                        help='choice of interaction')
    parser.add_argument('--n_theta', type=int, default=72,
                        help='number of segments in polar plot radially')
    parser.add_argument('--vr_n', type=int, default=10,
                        help='number of segments in polar plot linearly')

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

    logging.info('{dataset:>60s} |     N'.format(dataset=''))
    for dataset_file in train_dataset_names:
        filename = os.path.join(args.path,"train/train/real_data",dataset_file+".ndjson")
        logging.info('{dataset:>60s} | {N:>5}'.format(
            dataset=filename,
            N=sum(1 for _ in load_all(filename)),
        ))
    interaction_type = args.interaction_type
    trajectory_type  = args.trajectory_type

    for dataset_file in train_dataset_names:
        filename = os.path.join(args.path,"train/train/real_data",dataset_file+".ndjson")
        interaction_plots(filename, trajectory_type, interaction_type, args)


if __name__ == '__main__':
    main()
