import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense


class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims, fc2_dims):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    
    def call(self, state, action):
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)

        q = self.q(action_value)

        return q


class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims, fc2_dims, n_actions):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.mu = Dense(self.n_actions, activation='tanh') # Action boundry in range +/-1
    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)

        # If action range is bigger than +/-1, multiply by the corresponding value here.
        mu = self.mu(prob)

        return mu