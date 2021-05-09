import sys,os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import argparse
import time
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
print('[INF] Tensorflow version: ',tf.__version__)
tf.test.gpu_device_name()
import random
from datetime import datetime
random.seed(datetime.now())
import matplotlib.pyplot as plt

# Important imports
import path_prediction.batches_data
from path_prediction.training_utils import Experiment_Parameters

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../trajnet++/trajnetplusplustools')))
import trajnetplusplustools
import logging
import socket


def prepare_data(path, subset='/train/', sample=1.0):
    """ Prepares the train/val scenes and corresponding goals

    Parameters
    ----------
    subset: String ['/train/', '/val/']
        Determines the subset of data to be processed
    sample: Float (0.0, 1.0]
        Determines the ratio of data to be sampled
    goals: Bool
        If true, the goals of each track are extracted
        The corresponding goal file must be present in the 'goal_files' folder
        The name of the goal file must be the same as the name of the training file

    Returns
    -------
    all_scenes: List
        List of all processed scenes
    all_goals: Dictionary
        Dictionary of goals corresponding to each dataset file.
        None if 'goals' argument is False.
    """

    ## read goal files
    all_scenes = []

    ## List file names
    files = [f.split('.')[-2] for f in os.listdir(path + subset) if f.endswith('.ndjson')]
    ## Iterate over file names
    for file in files:
        reader = trajnetplusplustools.Reader(path + subset + file + '.ndjson', scene_type='paths')
        ## Necessary modification of train scene to add filename
        scene = [(file, s_id, s) for s_id, s in reader.scenes(sample=sample)]
        print("[INF] File ",file," for ",subset," with ",scene[0][1]," trajectories.")
        all_scenes += scene
    return all_scenes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=25, type=int, help='number of epochs')
    parser.add_argument('--save_every', default=5, type=int,help='frequency of saving model (in terms of epochs)')
    parser.add_argument('--obs_length', default=9, type=int,help='observation length')
    parser.add_argument('--pred_length', default=12, type=int,help='prediction length')
    parser.add_argument('--start_length', default=0, type=int,help='starting time step of encoding observation')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=1e-3, type=float,help='initial learning rate')
    parser.add_argument('--step_size', default=10, type=int,help='step_size of lr scheduler')
    parser.add_argument('-o', '--output', default=None,help='output file')
    parser.add_argument('--path', default='trajdata',help='glob expression for data files')
    parser.add_argument('--type', default='vanilla',choices=('vanilla', 'occupancy', 'directional', 'social', 'hiddenstatemlp', 's_att_fast','directionalmlp', 'nn', 'attentionmlp', 'nn_lstm', 'traj_pool', 'nmmp', 'dir_social'),help='type of interaction encoder')
    parser.add_argument('--sample', default=1.0, type=float,help='sample ratio when loading train/val scenes')

    ## Augmentations
    parser.add_argument('--augment', action='store_true',help='perform rotation augmentation')
    parser.add_argument('--normalize_scene', action='store_true',help='rotate scene so primary pedestrian moves northwards at end of observation')
    parser.add_argument('--augment_noise', action='store_true',help='flag to add noise to observations for robustness')

    ## Loading pre-trained models
    pretrain = parser.add_argument_group('pretraining')
    pretrain.add_argument('--load-state', default=None,help='load a pickled model state dictionary before training')
    pretrain.add_argument('--load-full-state', default=None, help='load a pickled full state dictionary before training')
    pretrain.add_argument('--nonstrict-load-state', default=None, help='load a pickled state dictionary before training')

    ## Sequence Encoder Hyperparameters
    hyperparameters = parser.add_argument_group('hyperparameters')
    hyperparameters.add_argument('--hidden-dim', type=int, default=128,help='LSTM hidden dimension')
    hyperparameters.add_argument('--coordinate-embedding-dim', type=int, default=64,help='coordinate embedding dimension')
    hyperparameters.add_argument('--pool_dim', type=int, default=256, help='output dimension of interaction vector')

    args = parser.parse_args()

    ## Fixed set of scenes if sampling
    if args.sample < 1.0:
        # TODO
        #torch.manual_seed("080819")
        random.seed(1)

    ## Define location to save trained model
    if not os.path.exists('OUTPUT_BLOCK/{}'.format(args.path)):
        os.makedirs('OUTPUT_BLOCK/{}'.format(args.path))
    args.output = 'OUTPUT_BLOCK/{}/vae_{}_{}.pkl'.format(args.path, args.type, args.output)

    # configure logging
    from pythonjsonlogger import jsonlogger
    if args.load_full_state:
        file_handler = logging.FileHandler(args.output + '.log', mode='a')
    else:
        file_handler = logging.FileHandler(args.output + '.log', mode='w')
    # TODO
    #file_handler.setFormatter(jsonlogger.JsonFormatter('(message) (levelname) (name) (asctime)'))
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler])
    logging.info({
            'type': 'process',
            'argv': sys.argv,
            'args': vars(args),
            'hostname': socket.gethostname(),
    })

    # refactor args for --load-state
    # loading a previously saved model
    args.load_state_strict = True
    if args.nonstrict_load_state:
        args.load_state = args.nonstrict_load_state
        args.load_state_strict = False
    if args.load_full_state:
        args.load_state = args.load_full_state

    ## Prepare data
    train_scenes = prepare_data(args.path, subset='/train/', sample=args.sample)
    ntrain = 0
    for scene_i, (filename, scene_id, paths) in enumerate(train_scenes):
        scene_start = time.time()
        ## Make new scene
        paths_xy = trajnetplusplustools.Reader.paths_to_xy(paths)
        ntrain += paths_xy.shape[1]
    val_scenes   = prepare_data(args.path, subset='/val/', sample=args.sample)
    nvals = 0
    for scene_i, (filename, scene_id, paths) in enumerate(val_scenes):
        scene_start = time.time()
        ## Make new scene
        paths_xy = trajnetplusplustools.Reader.paths_to_xy(paths)
        nvals += paths_xy.shape[1]
    print("[INF] Total number of training trajectories:",ntrain)
    print("[INF] Total number of validation trajectories:",nvals)

if __name__ == '__main__':
    main()
