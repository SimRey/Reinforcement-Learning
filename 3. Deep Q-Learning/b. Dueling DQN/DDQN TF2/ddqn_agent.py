import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, clone_model 
from tensorflow.keras.optimizers import Adam
from replay_buffer import ReplayBuffer


class DDQN(keras.Model):
    def __init__(self, n_actions, fc1_dims, fc2_dims):
        super(DDQN, self).__init__()
        self.dense1 = Dense(fc1_dims, activation='relu')
        self.dense2 = Dense(fc2_dims, activation='relu')
        self.V = Dense(1, activation=None) # Value layer --> single value
        self.A = Dense(n_actions, activation=None) # Action layer
    
    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)

        Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True)))
        return Q
    
    def advantage(self, state):
    
        x = self.dense1(state)
        x = self.dense2(x)
        A = self.A(x)

        return A




    

class Agent:
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, mem_size, batch_size, 
                 eps_min=0.01, eps_dec=5e-7, replace=100, fname='dqn_model.h5'):
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]

        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size)

        self.model = DDQN(n_actions, 256, 256)
        self.target = DDQN(n_actions, 256, 256)

        self.model.compile(optimizer=Adam(learning_rate=lr), loss="mse")

        self.model_file = fname




    def store_transition(self, state, action, reward, state_, done):
        """The following funtion is used to 'remember' the previously stated function in the replay buffer file"""

        self.memory.store_transition(state, action, reward, state_, done)


    def choose_action(self, observation):
        """This function performs an epsilon greedy action choice"""

        random_number = np.random.random()
        if random_number > self.epsilon:
            action = np.argmax(self.model.predict(np.array([observation])))
        else:
            action = np.random.choice(self.action_space)
        return action


    def replace_target_network(self):
        """Funtion used to replace the w-values in the target neural network with the w-values of the 
        model neural network, this replacement occurs every n steps"""

        if self.learn_step_counter > 0 and self.learn_step_counter % self.replace_target_cnt == 0:
            self.target.set_weights(self.model.get_weights())


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

        # Check if the target network needs to be updated
        self.replace_target_network()

        # Mini batch values
        states, actions, rewards, states_, dones = self.memory.mini_batch(self.batch_size)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        
        # Predict targets for all states from the sample
        targets = self.target.predict(np.array(states))
        
        # Predict Q-Values for all new states from the sample
        q_next = self.model.predict(np.array(states_))  

        # Replace the targets values with the according function
        targets[batch_index, actions] = rewards + self.gamma * np.max(q_next, axis=1)*(1 - np.array([dones]))
        
        # Fit the model based on the states and the updated targets for 1 epoch
        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0) 

        self.learn_step_counter += 1
        self.decrement_epsilon()


    def save_model(self):
        self.model.save(self.model_file)
    
    def load_model(self):
        self.model = load_model(self.model_file)