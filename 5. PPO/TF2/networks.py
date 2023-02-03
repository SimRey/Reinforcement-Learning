import tensorflow as tf 
import tensorflow_probability as tfp 
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

tfd = tfp.distributions

class ActorNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = Dense(self.fc1_dims, activation='tanh')
        self.fc2 = Dense(self.fc2_dims, activation='tanh')
        self.alpha = Dense(self.n_actions, activation='relu')
        self.beta = Dense(self.n_actions, activation='relu')
       

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        alpha = self.alpha(x) + 1.0
        beta = self.beta(x) + 1.0
        dist = tfd.Beta(alpha, beta)
        return dist



class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=256, fc2_dims=256):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = Dense(self.fc1_dims, activation='tanh')
        self.fc2 = Dense(self.fc2_dims, activation='tanh')
        self.v = Dense(1, activation=None)
        

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        v = self.v(x)

        return v

    