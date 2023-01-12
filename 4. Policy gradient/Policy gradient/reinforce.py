import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential 
from tensorflow.keras.optimizers import Adam

class PolicyGradientNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims, fc2_dims, fc3_dims, fc4_dims):
        super(PolicyGradientNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.n_actions = n_actions

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.fc3 = Dense(self.fc3_dims, activation='relu')
        self.fc4 = Dense(self.fc4_dims, activation='relu')
        self.pi = Dense(self.n_actions, activation='softmax') # Policy layer
    
    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        pi = self.pi(x)

        return pi # Returns policy

    
class Agent:
    def __init__(self, gamma, lr, n_actions, fname='reinforce'):
        
        self.gamma = gamma
        self.lr = lr
        self.n_actions = n_actions
        self.state_memory = []
        self.reward_memory = []
        self.action_memory = []

        self.policy = PolicyGradientNetwork(self.n_actions, 512, 256, 128, 72)
        self.policy.compile(optimizer=Adam(learning_rate=lr))

        self.model_file = fname


    def store_transition(self, state, action, reward):
        """The following funtion is used to store the generated state, action and reward"""
        self.state_memory.append(state)
        self.reward_memory.append(reward)
        self.action_memory.append(action)


    def choose_action(self, observation):
        """This function choses an action using the policy"""

        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        probs = self.policy(state)
        action_probs = tfp.distributions.Categorical(probs=probs)
        action = action_probs.sample()

        return action.numpy()[0] # To interact correct with gym environment

    def learn(self):
        """Funtion that describes the learning process of the RL-Agent"""

        actions = tf.convert_to_tensor(self.action_memory, dtype=tf.float32)
        rewards = np.array(self.reward_memory, dtype=np.float32)

        # Monte-Carlo reward function
        G = np.zeros_like(rewards,)
        for t in range(len(rewards)):
            G_sum = 0
            discount = 1
            for k in range(t, len(rewards)):
                G_sum += rewards[k]*discount
                discount += self.gamma
            G[t] = G_sum
        
        # Gradient calculations
        with tf.GradientTape() as tape:
            loss = 0

            for idx, (g, state) in enumerate(zip(G, self.state_memory)):
                state = tf.convert_to_tensor([state], dtype=tf.float32)
                probs = self.policy(state)
                action_probs = tfp.distributions.Categorical(probs=probs)
                log_prob = action_probs.log_prob(actions[idx])
                loss += -g*log_prob
        
        params = self.policy.trainable_variables
        grads = tape.gradient(loss, params)
        self.policy.optimizer.apply_gradients(zip(grads, params))

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

    def save_model(self):
        self.policy.save(self.model_file)
    
    def load_model(self):
        self.policy = keras.models.load_model(self.model_file)