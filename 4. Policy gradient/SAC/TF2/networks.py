import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense


class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=256, fc2_dims=256):
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


class ValueNetwork(keras.Model):
    def __init__(self, fc1_dims=256, fc2_dims=256):
        super(ValueNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.v = Dense(1, activation=None)

    def call(self, state):
        state_value = self.fc1(state)
        state_value = self.fc2(state_value)

        v = self.v(state_value)

        return v


class ActorNetwork(keras.Model):
    def __init__(self, n_actions, layer_size=256, log_std_min=-20, log_std_max=2):
        super(ActorNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.layer1 = Dense(layer_size, activation='relu')
        self.layer2 = Dense(layer_size, activation='relu')
        self.layer3 = Dense(layer_size, activation='relu')

        self.mean = Dense(n_actions, activation='linear')
        self.log_std = Dense(n_actions, activation='linear')


    def call(self, inputs):
        out = self.layer1(inputs)
        out = self.layer2(out)
        out = self.layer3(out)

        mean = self.mean(out)
        log_std = self.log_std(out)
        log_std = tf.keras.backend.clip(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std