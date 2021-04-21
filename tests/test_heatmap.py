import sys,os
sys.path.append('../lib')
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
print('Tensorflow version: ',tf.__version__)
tf.test.gpu_device_name()
import random
from datetime import datetime
random.seed(datetime.now())
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
import numpy as np
import seaborn as sns
from datasets_utils import setup_loo_experiment, get_testing_batch
from training_and_testing import Experiment_Parameters
from model import TrajectoryEncoderDecoder, Model_Parameters
import batches_data
from obstacles import image_to_world_xy

# Load the default parameters
experiment_parameters = Experiment_Parameters(add_social=False,add_kp=False,obstacles=False)
dataset_dir       = "../../datasets/"
dataset_paths     = [dataset_dir+'eth-hotel',dataset_dir+'eth-univ',dataset_dir+'ucy-zara01']
# Load the dataset and perform the split
idTest = 2
training_data,validation_data,test_data,test_homography = setup_loo_experiment('ETH_UCY',dataset_paths,idTest,experiment_parameters,use_pickled_data=False,pickle_dir='./')
test_homography = np.linalg.inv(test_homography)

# Model parameters
model_parameters = Model_Parameters()
# Get the necessary data
test_data        = batches_data.Dataset(test_data, model_parameters)

# Plot ramdomly a subset of the training data (spatial data only)
plt.subplots(1,1,figsize=(10,10))
plt.subplot(1,1,1)
plt.axis('equal')
# Plot some of the testing data
batch, test_bckgd = get_testing_batch(test_data,dataset_paths[idTest])
for (o,p) in zip(batch["obs_traj"],batch["pred_traj"]):
    # Observations
    oi= image_to_world_xy(o, test_homography,flip=False)
    plt.plot(oi[:,0],oi[:,1],color='red')
    # Prediction targets
    p = np.concatenate((o[-1].reshape(1,2),p))
    p= image_to_world_xy(p, test_homography,flip=False)
    #plt.plot(x,y,color='blue')
    hmax = sns.kdeplot(p[:,0],p[:,1], cmap="Blues", shade=True, bw=.5)
    hmax.collections[0].set_alpha(0)
plt.imshow(test_bckgd, zorder=0)
plt.show()
plt.title("Samples of trajectories ")
plt.show

# Custom it with the same argument as 1D density plot
