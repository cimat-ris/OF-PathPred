import sys,os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
print('[INF] Tensorflow version: ',tf.__version__)
tf.test.gpu_device_name()
import random
from datetime import datetime
random.seed(datetime.now())
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
import numpy as np
import seaborn as sns
from path_prediction.testing_utils  import get_testing_batch
from path_prediction.datasets_utils import setup_loo_experiment
from path_prediction.training_utils import Experiment_Parameters
from path_prediction.model import TrajectoryEncoderDecoder, Model_Parameters
from path_prediction.obstacles import image_to_world_xy

# Load the default parameters
experiment_parameters = Experiment_Parameters(add_social=False,add_kp=False,obstacles=False)
dataset_dir   = "datasets/"
dataset_names = ['eth-hotel','eth-univ','ucy-zara01','ucy-zara02','ucy-univ']

# Load the dataset and perform the split
idTest = 2
training_data,validation_data,test_data,test_homography = setup_loo_experiment('ETH_UCY',dataset_dir,dataset_names,idTest,experiment_parameters,use_pickled_data=False)

# Model parameters
model_parameters = Model_Parameters()
# Get the necessary data
test_data        = tf.data.Dataset.from_tensor_slices(test_data)
# Plot ramdomly a subset of the training data (spatial data only)
plt.subplots(1,1,figsize=(10,10))
plt.subplot(1,1,1)
plt.axis('equal')
# Plot some of the testing data
batch, test_bckgd = get_testing_batch(test_data,dataset_dir+dataset_names[idTest])

test_homography = np.linalg.inv(test_homography)

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
