import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probabil

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, clone_model 
from tensorflow.keras.optimizers import Adam
from replay_buffer import ReplayBuffer

class PolicyGradientNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims, fc2_dims):
        super(PolicyGradientNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.pi = Dense(self.n_actions, activation='softmax') # Policy layer
    
    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)

        pi = self.pi(x)

        return pi # Returns policy

    

class Agent:
    def __init__(self, gamma, lr, n_actions, input_dims, mem_size, batch_size, 
                 fname='dqn_model.h5'):
        
        self.gamma = gamma
        self.lr = lr
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]

        self.input_dims = input_dims
        self.batch_size = batch_size
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size)

        self.policy = policy_NN(input_dims, n_actions, 256, 256)
        self.policy.compile(optimizer=Adam(learning_rate=lr), loss="mse")

        self.model_file = fname


    def store_transition(self, state, action, reward, state_, done):
        """The following funtion is used to 'remember' the previously stated function in the replay buffer file"""

        self.memory.store_transition(state, action, reward, state_, done)


    def choose_action(self, observation):
        """This function choses an action using the policy"""

        probs = self.policy.predict(np.array([observation]))
        action_probs = tfp.distributions.Categorical(probs=probs)
        action = action_probs.sample()

        return action




    def decrement_epsilon(self):
        """Funtion used to compute the decrement of epsilon"""
        if self.epsilon > self.eps_min:
            self.epsilon -= self.eps_dec
        else:
            self.epsilon = self.eps_min


    def learn(self):
        """Funtion that describes the learning process of the RL-Agent"""

        if len(self.memory.replay_buffer) < self.batch_size:
            return

        # Mini batch values
        states, actions, rewards, states_, dones = self.memory.mini_batch(self.batch_size)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # Check if the target network needs to be updated
        self.replace_target_network()

        # Predict Q-Values for all new states from the sample --> target network
        q_next = self.target.predict(np.array(states_))

        # Predict targets for all states from the sample to perform update --> model
        q_target = self.model.predict(np.array(states))

        # Replace the targets values with the according function
        q_target[batch_index, actions] = rewards + self.gamma * np.max(q_next, axis=1)*(1 - np.array([dones]))

        # Fit the model based on the states and the updated targets for 1 epoch
        self.model.fit(np.array(states), np.array(q_target), epochs=1, verbose=0)


        self.learn_step_counter += 1
        self.decrement_epsilon()


    def save_model(self):
        self.model.save(self.model_file)
    
    def load_model(self):
        self.model = keras.model.load_model(self.model_file)