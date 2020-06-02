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
from keras.models import Model, Sequential, save_model, load_model
import math
from sklearn.metrics import mean_squared_error
import random
from sklearn.preprocessing import MinMaxScaler
import h5py
import tensorflow as tf
print('[INF] TF Version: '+tf.__version__)

from lib.preprocess import *
from lib.evaluation import *
from lib.sequence_preparation import *
from lib.models import *


datasets  = [0]
data_dirs = ['../data1/pets','../data1/ucy/univ','../data1/ucy/zara/zara02','../data1/eth/hotel','../data1/ucy/zara/zara01']
used_data_dirs = [data_dirs[x] for x in datasets]
# Directory with the preprocessed pickle files
data_dir = '../data1'
#Defina la ruta del archivo en el que deben almacenarse los datos.
preprocessed_file = os.path.join(data_dir, "datos_limpios_.cpkl")
# Para pets si se usa pixel_pos.csv es el de framerate 7.5
# Para pets si se usa pixel_pos_2.csv es el de framerate 3.75
data_filename ='pixel_pos.csv'
data            = preprocess(used_data_dirs,data_filename,preprocessed_file)
# Reload it
data_p,number_p = load_preprocessed(preprocessed_file,12,1)
print("[INF] Number of samples "+str(number_p))



# ## Trajectory visualization
import random
color_names = ["r","crimson" ,"g", "b","c","m","y","lightcoral", "peachpuff","grey","springgreen" ,"fuchsia","violet","teal","seagreen","lime","yellow","coral","aquamarine","hotpink"]

for i in range(len(data_p)):
    plt.plot(data_p[i][:,0],data_p[i][:,1],color=color_names[i])
plt.title("Full trajectories in PETS-2009")
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.savefig("PETS2009-alltrajctories.pdf")
plt.show()


# ## Splitting the data
def split_data_idgroups(combination,intervals_ids,data):
    # Form different combinations of training/testing sets
    training_set = []
    for i in range(len(combination)-1):
        for j in intervals_ids[combination[i]]:
            training_set.append(data[j])
    testing_set = []
    for i in intervals_ids[combination[4]]:
        testing_set.append(data[i])
    return training_set,testing_set

split_mode   = 1
total_length = len(data_p)

if split_mode==0:
    indices      = range(total_length)
    intervals_ids        = []
    # Pedestrians are split in groups of 4
    for i in range(0,total_length,4):
        intervals_ids.append(indices[i:i+4])

    # TODO: generalize?
    combinations=[(0,1,2,3,4),(0,1,2,4,3),(0,1,3,4,2),(0,2,3,4,1),(1,2,3,4,0)]
    # Generate train/test
    train1,test1 = split_data_idgroups(combinations[0],intervals_ids,data_p)
    train2,test2 = split_data_idgroups(combinations[1],intervals_ids,data_p)
    train3,test3 = split_data_idgroups(combinations[2],intervals_ids,data_p)
    train4,test4 = split_data_idgroups(combinations[3],intervals_ids,data_p)
    train5,test5 = split_data_idgroups(combinations[4],intervals_ids,data_p)

elif split_mode==1:
    random.seed(0)
    indices      = arange(total_length)
    data_p = np.array(data_p)
    random.shuffle(indices)
    training_size = int(total_length * 0.80)
    testing_size  = total_length-training_size

    train1 = data_p[indices[0:training_size]]
    test1  = data_p[indices[training_size:]]

    print("[INF] Number of pedestrians "+str(total_length))
    print("[INF] Training with " ,len(train1))
    print("[INF] Testing with " ,len(test1))


# In[84]:


plt.subplot(121)
plt.plot(train1[0][:,0],train1[0][:,1])
if split_mode==0:
    plt.subplot(122)
    plt.plot(train5[0][:,0],train5[0][:,1])


# In[109]:


length_obs         = 8
representation_mode= 'dxdy'

if representation_mode=='xy':
    trainX,trainY = split_sequence_training_xy(length_obs,train1)
if representation_mode=='dxdy':
    trainX,trainY = split_sequence_training_dxdy(length_obs,train1)
if representation_mode=='lineardev':
    trainX,trainY = split_sequence_training_lineardev(length_obs,train1)


trainX = np.reshape(trainX, (trainX.shape[0],trainX.shape[1],trainX.shape[2]))
data_shape = trainX.shape[1:]
print('[INF] Shape of training data ',np.shape(trainX))


# Build and train the network
model = SingleStepPrediction(representation_mode)
model.training_loop(trainX,trainY)

# Plot some samples of the prediction, on the training dataset
model.plot_prediction_samples(trainX,trainY)

# Save the model
model.save_weights('models/simple-{}.ckpt'.format(representation_mode))
# Load the model
model.load_weights('models/simple-{}.ckpt'.format(representation_mode))

# Evaluate the errors
model.evaluate_testing_set(test1,8,4,True)
# Evaluate on the whole dataset
model.evaluate_testing_set_start(data_p,8,4)


# ## Qualitative evaluation

# In[129]:


cruce = []
cruce.append(data_p[3][252:264,:])
cruce.append(data_p[4][135:147,:])

paralelos = []
paralelos.append(data_p[2][1:13])
paralelos.append(data_p[3][20:32])

inverso = []
inverso.append(data_p[4][167:179])
inverso.append(data_p[6][15:27])

"""
Para graficar los resultados cualitativos se escoge el mejor modelo de los
5,  para pets con framerate 7.5
"""
band = 0
if(band==0):
    p,v = sample_en_pixeles_cualitativamente(model,cruce,8,4,representation_mode)
    name = "cruce_absoluto_ing.pdf"
elif(band==1):
    p,v = sample_en_pixeles_cualitativamente(model,paralelos,8,4,representation_mode)
    name = "paralelos_absoluto_ing.pdf"
else:
    p,v = sample_en_pixeles_cualitativamente(model,inverso,8,4,representation_mode)
    name = "inverso_absoluto_ing.pdf"
plot_qualitative(p,v,name)


# In[ ]:
