import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import initializers
from tensorflow.keras.utils import plot_model
from lib.evaluation import *
import matplotlib.pyplot as plt

class SingleStepPrediction(tf.keras.Model):
    # Constructor, the layers are defined here
    def __init__(self,mode='xy',hidden_state=10):
        super(SingleStepPrediction, self).__init__(name='SingleStepPrediction')
        self.output_representation_mode = mode
        self.rnn1 = LSTM(hidden_state, return_sequences=True, name='lstm1')
        self.rnn2 = LSTM(hidden_state, name='lstm2')
        self.regression= Dense(2)

    # Build the model
    def build(self, batch_input_shape):
        super().build(batch_input_shape)

    # Forward pass
    def call(self, inputs):
        x = self.rnn1(inputs)
        x = self.rnn2(x)
        return self.regression(x)

    # Training loop
    def training_loop(self,trainX,trainY, epochs=250, batch_size=64):
        self.compile(optimizer=optimizers.RMSprop(lr = 0.01, decay=1e-2), loss='logcosh',metrics=['mse'])
        history= self.fit(trainX, trainY, epochs=250, batch_size=64, verbose=2)
        self.summary()

        history_dict= history.history
        history_dict.keys()
        acc  = history.history['mean_squared_error']
        loss = history.history['loss']
        #val_acc = history.history['val_mean_squared_error']s
        #val_loss = history.history['val_loss']

        # Plot results
        epochs = range(1, len(loss)+1)
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(epochs, loss, 'b', label='Training loss')
        #plt.plot(epochs, val_loss, 'g', label='Validation loss')
        plt.title('Training loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(epochs, acc, 'r', label='Training mse')
        #plt.plot(epochs, val_acc, 'g', label='Validation mse')
        plt.title('Training mse')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.legend()
        plt.show()

    # Plot prediction results
    def plot_prediction_samples(self,dataX,dataY):
        # Select randomly 9 trajectories
        rnds = np.random.randint(0, np.shape(dataX)[0], size=9)
        plt.subplots(3,3,figsize=(15,15))
        for i in range(9):
            plt.subplot(3,3,1+i)
            traj_obsr = np.reshape(dataX[rnds[i]], (1,dataX[rnds[i]].shape[0],dataX[rnds[i]].shape[1]) )
            # Applies the model to produce the next point
            next_point= self.predict(traj_obsr)
            plt.plot(dataX[rnds[i]][:,0],dataX[rnds[i]][:,1])
            if self.output_representation_mode=='xy':
                plt.plot(next_point[0][0],next_point[0][1],'go')
                plt.plot(dataY[rnds[i]][0],dataY[rnds[i]][1],'r+')
            if self.output_representation_mode=='dxdy':
                plt.plot(dataX[rnds[i]][-1,0]+next_point[0][0], dataX[rnds[i]][-1,1]+next_point[0][1],'go')
                plt.plot(dataX[rnds[i]][-1,0]+dataY[rnds[i]][0],dataX[rnds[i]][-1,1]+dataY[rnds[i]][1],'r+')
            if self.output_representation_mode=='lineardev':
                x0,y0,vx,vy   = linear_lsq_model(dataX[rnds[i]][:,0],dataX[rnds[i]][:,1])
                x_pred_linear = x0+vx*(len(dataX[rnds[i]][:,0])+1)
                y_pred_linear = y0+vy*(len(dataX[rnds[i]][:,1])+1)
                plt.plot(x_pred_linear+next_point[0][0], y_pred_linear+next_point[0][1],'go')
                plt.plot(x_pred_linear+dataY[rnds[i]][0],y_pred_linear+dataY[rnds[i]][1],'r+')
        plt.axis('equal')

    # Predict several steps ahead with this model
    def predict_steps(self,X,seq_length_pred):
        traj_obs  = X
        traj_pred = X

        for j in range(seq_length_pred):
            traj_obsr  = np.reshape(traj_obs, (1,traj_obs.shape[0],traj_obs.shape[1]) )
            # Applies the model
            next_point = self.predict(traj_obsr)
            if self.output_representation_mode=='dxdy':
                next_point=next_point+traj_obs[-1]
            if self.output_representation_mode=='lineardev':
                x0,y0,vx,vy   = linear_lsq_model(traj_obs[:,0],traj_obs[:,1])
                x_pred_linear = x0+vx*(len(traj_obs[:,0])+1)
                y_pred_linear = y0+vy*(len(traj_obs[:,1])+1)
                next_point=next_point+[x_pred_linear,y_pred_linear]
            traj_obs   = np.concatenate((traj_obs[1:len(traj_obs)], next_point), axis = 0)
            traj_pred  = np.concatenate((traj_pred, next_point), axis = 0)
        return traj_obs,traj_pred


    # Takes a testing set and evaluates errors on it
    def evaluate(self, testX, testY, seq_length_obs, seq_length_pred, pixels=False):
        # Observations and targets, by splitting the trajectories of the testing set
        total_ade    = 0.0
        total_fde    = 0.0
        all_traj_pred= []
        all_traj_gt  = []
        # Iterate over the splitted testing data
        for i in range(len(testX)):
            traj_obs,traj_pred = self.predict_steps(testX[i],seq_length_pred)
            # Evaluate the difference
            if pixels:
                traj_pred = np.column_stack((768*traj_pred[:,0],576*traj_pred[:,1]))
                traj_gt   = np.column_stack((768*testY[i][:,0],576*testY[i][:,1]))
            else:
                traj_gt = testY
            total_ade += evaluate_ade(traj_pred, traj_gt, seq_length_obs)
            total_fde += evaluate_fde(traj_pred, traj_gt, seq_length_obs)
            # Keep GT and predicted trajectories in pixels
            all_traj_pred.append(traj_pred)
            all_traj_gt.append(traj_gt)

        error_ade = total_ade/len(testX)
        error_fde = total_fde/len(testX)
        if pixels:
            print('---------Error (pixels)--------')
        else:
            print('---------Error (normalized coordinates)--------')
        print('[RES] ADE: ',error_ade)
        print('[RES] FDE: ',error_fde)
        return all_traj_pred, all_traj_gt

    # Plot predictions
    def predict_and_plot(self, testX, testY, seq_length_obs, seq_length_pred, pixels=False):
        # Just the starting parts of the sequence
        plt.figure(figsize=(18,15))
        color_names       = ["r","crimson" ,"g", "b","c","m","y","lightcoral", "peachpuff","grey","springgreen" ,"fuchsia","violet","teal","seagreen","lime","yellow","coral","aquamarine","hotpink"]
        plt.subplot(1,1,1)
        # For all the subsequences
        for i in range(len(testX)):
            traj_obs,traj_pred = self.predict_steps(testX[i],seq_length_pred)
            plt.plot(testY[i][0:8,0],testY[i][0:8,1],'*--',color=color_names[19-i],label = 'Observed')
            plt.plot(testY[i][7:,0],testY[i][7:,1],'--',color=color_names[i],label = 'GT')
            plt.plot(traj_pred[seq_length_obs-1:,0],traj_pred[seq_length_obs-1:,1],'-',color=color_names[19-i],label = 'Predicted')
            plt.axis('equal')
        plt.title("Predicting 4 positions with LTM-X-Y")
        plt.xlabel('x-coordinate')
        plt.ylabel('y-coordinate')
        #plt.savefig("4predichas.pdf")
        plt.show()
