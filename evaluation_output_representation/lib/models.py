import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import initializers
from tensorflow.keras.utils import plot_model

class SingleStepPrediction(tf.keras.Model):
    # Constructor, the layers are defined here
    def __init__(self,hidden_state=10):
        super(SingleStepPrediction, self).__init__(name='SingleStepPrediction')
        self.rnn1 = LSTM(hidden_state, return_sequences=True, name='lstm1')
        self.rnn2 = LSTM(hidden_state, name='lstm2')
        self.regression= Dense(2)

    def build(self, batch_input_shape):
        super().build(batch_input_shape)

    # Forward pass
    def call(self, inputs):
        x = self.rnn1(inputs)
        x = self.rnn2(x)
        return self.regression(x)
