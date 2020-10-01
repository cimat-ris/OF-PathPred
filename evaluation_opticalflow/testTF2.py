# Imports
import sys,os
sys.path.append('./lib')
import math,numpy as np
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
print('Tensorflow version: ',tf.__version__)
tf.test.gpu_device_name()
# Important imports
from process_file import process_file
import batches_data
from training_and_testing import Trainer,Tester,Experiment_Parameters
import matplotlib.pyplot as plt
from model import TrajectoryEncoderDecoder, Model_Parameters

model_parameters = Model_Parameters(train_num_examples=1,add_kp=False,add_social=True)
# x is NxTx2
x = tf.ones((3,10,2))
s = tf.ones((3,10,20))
tj_enc_dec = TrajectoryEncoderDecoder(model_parameters)
xp     = tj_enc_dec(x,s)
print(xp)
