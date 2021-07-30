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

import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs_length', default=8, type=int,help='observation length')
    parser.add_argument('--pred_length', default=12, type=int,help='prediction length')
    parser.add_argument('--path', default='datasets/trajnetplusplus',help='glob expression for data files')
    parser.add_argument('--log_level',type=int, default=20,help='Log level (default: 20)')
    parser.add_argument('--log_file',default='',help='Log file (default: standard output)')
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

    for dataset_file in train_dataset_names:
        filename = os.path.join(args.path,"train/train/real_data",dataset_file+".ndjson")
        dataset_plots(filename, obs_length=args.obs_length)


if __name__ == '__main__':
    main()
