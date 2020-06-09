#!/usr/bin/env python
# coding: utf-8

# This script performs trajectory predicton the PETS-S2L1 dataset.
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pylab import *
from scipy. ndimage import filters
import os
import math
from sklearn.metrics import mean_squared_error
import random
from sklearn.preprocessing import MinMaxScaler
import h5py
import tensorflow as tf
print('[INF] TF Version: '+tf.__version__)

import sys
sys.path.append('../evaluation_output_representation/lib')
from lib.preprocess import *
from lib.sequence_preparation import *
from lib.models import *

################################################################################################################
datasets  = [0]
data_dirs = ['../data1/pets','../data1/ucy/univ','../data1/ucy/zara/zara02','../data1/eth/hotel','../data1/ucy/zara/zara01']
used_data_dirs = [data_dirs[x] for x in datasets]
# Directory with the preprocessed pickle files
data_dir = '../data1'
# Defina la ruta del archivo en el que deben almacenarse los datos.
preprocessed_file = os.path.join(data_dir, "datos_limpios_.cpkl")
# Para pets si se usa pixel_pos.csv es el de framerate 7.5
# Para pets si se usa pixel_pos_2.csv es el de framerate 3.75
data_filename ='pixel_pos.csv'
data            = preprocess(used_data_dirs,data_filename,preprocessed_file)
# Reload it
data_p,number_p = load_preprocessed(preprocessed_file,12,1)
print("[INF] Number of samples "+str(number_p))
# Plot the trajectories
plot_trajectories(data_p)


################################################################################################################
# Parameters for defining observations/predictions
split_mode         = 1
length_obs         = 8
length_pred        = 4
representation_mode='dxdy'
id_train           = 1
# Generate training/testing datasets
trainX,trainY,testX,testY = split_data(data_p,split_mode,length_obs,length_pred,representation_mode,id_train)
allX,allY                 = split_sequence_start_testing(data_p,length_obs,length_pred,representation_mode)

data_shape = trainX.shape[1:]
print('[INF] Shape of training data ',np.shape(trainX))


################################################################################################################
# Build and train the network
model = SingleStepPrediction(representation_mode)
model.training_loop(trainX,trainY,epochs=350)

# Plot some samples of the prediction, on the training dataset
if representation_mode!='only_displacement':
    model.plot_prediction_samples(trainX,trainY)

# Save the model
model.save_weights('models/simple-{}.ckpt'.format(representation_mode))
# Load the model
model.load_weights('models/simple-{}.ckpt'.format(representation_mode))


################################################################################################################
# Evaluate the errors
model.evaluate(testX,testY,length_obs,length_pred,True)
# Evaluate on the whole dataset
model.predict_and_plot(allX,allY,length_obs,length_pred)


## Qualitative evaluation
cruce     = [data_p[3][252:264,:],data_p[4][135:147,:]]
paralelos = [data_p[2][1:13],data_p[3][20:32]]
inverso   = [data_p[4][167:179],data_p[6][15:27]]

# First example
testX,testY  = split_sequence_testing(cruce,length_obs,length_pred,representation_mode)
p,v = model.evaluate(testX,testY,length_obs,length_pred,True)
name = "cruce_absoluto_ing.pdf"
plot_qualitative(p,v,name)

# Second example
testX,testY  = split_sequence_testing(paralelos,length_obs,length_pred,representation_mode)
p,v = model.evaluate(testX,testY,length_obs,length_pred,True)
name = "paralelos_absoluto_ing.pdf"
plot_qualitative(p,v,name)

# Third example
testX,testY  = split_sequence_testing(inverso,length_obs,length_pred,representation_mode)
p,v = model.evaluate(testX,testY,length_obs,length_pred,True)
name = "inverso_absoluto_ing.pdf"
plot_qualitative(p,v,name)
