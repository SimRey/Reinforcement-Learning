import tensorflow.keras as keras
from tensorflow.keras.layers import Dense


class DeepQNetwork(keras.Model):
    def __init__(self, input_dims, n_actions, fc1_dims, fc2_dims):
        super(DeepQNetwork, self).__init__()
        
        self.fc1 = Dense(fc1_dims, input_shape=(input_dims,), activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.fc3 = Dense(n_actions, activation=None)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

