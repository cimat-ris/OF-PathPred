import sys,os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import argparse
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


def prepare_data(path, subset='/train/', sample=1.0, goals=True):
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
    all_goals = {}
    all_scenes = []

    ## List file names
    files = [f.split('.')[-2] for f in os.listdir(path + subset) if f.endswith('.ndjson')]
    ## Iterate over file names
    for file in files:
        print("Reading file ",file," for ",subset)
        reader = trajnetplusplustools.Reader(path + subset + file + '.ndjson', scene_type='paths')
        ## Necessary modification of train scene to add filename
        scene = [(file, s_id, s) for s_id, s in reader.scenes(sample=sample)]
        print("")
        print(scene[0])
        if goals:
            goal_dict = pickle.load(open('goal_files/' + subset + file +'.pkl', "rb"))
            ## Get goals corresponding to train scene
            all_goals[file] = {s_id: [goal_dict[path[0].pedestrian] for path in s] for _, s_id, s in scene}
        all_scenes += scene

    if goals:
        return all_scenes, all_goals
    return all_scenes, None

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
    parser.add_argument('--disable-cuda', action='store_true',help='disable CUDA')
    parser.add_argument('--path', default='trajdata',help='glob expression for data files')
    parser.add_argument('--goals', action='store_true',help='flag to consider goals of pedestrians')
    parser.add_argument('--loss', default='pred', choices=('L2', 'pred'),help='loss objective, L2 loss (L2) and Gaussian loss (pred)')
    parser.add_argument('--type', default='vanilla',choices=('vanilla', 'occupancy', 'directional', 'social', 'hiddenstatemlp', 's_att_fast','directionalmlp', 'nn', 'attentionmlp', 'nn_lstm', 'traj_pool', 'nmmp', 'dir_social'),help='type of interaction encoder')
    parser.add_argument('--sample', default=1.0, type=float,help='sample ratio when loading train/val scenes')

    ## Augmentations
    parser.add_argument('--augment', action='store_true',help='perform rotation augmentation')
    parser.add_argument('--normalize_scene', action='store_true',help='rotate scene so primary pedestrian moves northwards at end of observation')
    parser.add_argument('--augment_noise', action='store_true',help='flag to add noise to observations for robustness')
    parser.add_argument('--obs_dropout', action='store_true',help='perform observation length dropout')

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
    hyperparameters.add_argument('--goal_dim', type=int, default=64, help='goal embedding dimension')

    ## Grid-based pooling
    hyperparameters.add_argument('--cell_side', type=float, default=0.6, help='cell size of real world (in m) for grid-based pooling')
    hyperparameters.add_argument('--n', type=int, default=12, help='number of cells per side for grid-based pooling')
    hyperparameters.add_argument('--layer_dims', type=int, nargs='*', default=[512], help='interaction module layer dims for gridbased pooling')
    hyperparameters.add_argument('--embedding_arch', default='one_layer',help='interaction encoding arch for gridbased pooling')
    hyperparameters.add_argument('--pool_constant', default=0, type=int, help='background value (when cell empty) of gridbased pooling')
    hyperparameters.add_argument('--norm_pool', action='store_true', help='normalize the scene along direction of movement during grid-based pooling')
    hyperparameters.add_argument('--front', action='store_true', help='flag to only consider pedestrian in front during grid-based pooling')
    hyperparameters.add_argument('--latent_dim', type=int, default=16, help='latent dimension of encoding hidden dimension during social pooling')
    hyperparameters.add_argument('--norm', default=0, type=int, help='normalization scheme for input batch during grid-based pooling')

    ## Non-Grid-based pooling
    hyperparameters.add_argument('--no_vel', action='store_true', help='flag to not consider relative velocity of neighbours')
    hyperparameters.add_argument('--spatial_dim', type=int, default=32, help='embedding dimension for relative position')
    hyperparameters.add_argument('--vel_dim', type=int, default=32,help='embedding dimension for relative velocity')
    hyperparameters.add_argument('--neigh', default=4, type=int, help='number of nearest neighbours to consider')
    hyperparameters.add_argument('--mp_iters', default=5, type=int,help='message passing iterations in NMMP')

    ## VAE-Specific Parameters
    hyperparameters.add_argument('--alpha_kld', type=float, default=1.0, help='KLD loss weight')
    hyperparameters.add_argument('--k', type=int, default=1, help='number of samples for reconstruction loss')
    hyperparameters.add_argument('--desire', action='store_true', help='flag to use kld version of DESIRE')
    hyperparameters.add_argument('--noise_dim', type=int, default=64, help='noise dim of VAE')
    args = parser.parse_args()

    ## Fixed set of scenes if sampling
    if args.sample < 1.0:
        # TODO
        #torch.manual_seed("080819")
        random.seed(1)

    ## Define location to save trained model
    if not os.path.exists('OUTPUT_BLOCK/{}'.format(args.path)):
        os.makedirs('OUTPUT_BLOCK/{}'.format(args.path))
    if args.goals:
        args.output = 'OUTPUT_BLOCK/{}/vae_goals_{}_{}.pkl'.format(args.path, args.type, args.output)
    else:
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

    # add args.device
    # TODO
    # args.device = torch.device('cpu')
    # if not args.disable_cuda and torch.cuda.is_available():
    #     args.device = torch.device('cuda')

    args.path = 'DATA_BLOCK/' + args.path
    ## Prepare data
    train_scenes, train_goals = prepare_data(args.path, subset='/train/', sample=args.sample, goals=args.goals)
    val_scenes, val_goals = prepare_data(args.path, subset='/val/', sample=args.sample, goals=args.goals)


if __name__ == '__main__':
    main()
