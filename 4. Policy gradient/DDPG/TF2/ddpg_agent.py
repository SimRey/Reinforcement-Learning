import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from replay_buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork
from noise import OUActionNoise


class Agent:
    def __init__(self, n_actions,
        alpha=0.001, # Learning rate actor network
        beta=0.002, # Learning rate critic network (can be higher than actor network)
        env=None, gamma=0.99, mem_size=50000, tau=0.005, fc1_dims=400, fc2_dims=300, 
        batch_size=64, noise=0.1, chkpt_dir='models/ddpg/'):

        self.n_actions = n_actions
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mem_size = mem_size
        self.tau = tau
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.batch_size = batch_size
        self.noise = noise
        self.chkpt_dir = chkpt_dir

        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        self.noise = OUActionNoise(mu=np.zeros(self.n_actions,))
        self.memory = ReplayBuffer(self.mem_size)        

        self.actor = ActorNetwork(n_actions=self.n_actions, fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims)
        self.critic = CriticNetwork(fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims)

        self.target_actor = ActorNetwork(n_actions=self.n_actions, fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims)
        self.target_critic = CriticNetwork(fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims)

        self.actor.compile(optimizer=Adam(learning_rate=self.alpha))
        self.critic.compile(optimizer=Adam(learning_rate=self.beta))

        # No training or update will be used, however the compilation needs to be done to state the learning rate
        self.target_actor.compile(optimizer=Adam(learning_rate=self.alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=self.beta))

        self.update_network_parameters(tau=1) # Initialize to same values

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save(self.chkpt_dir+'actor')
        self.target_actor.save(self.chkpt_dir+'target_actor')
        self.critic.save(self.chkpt_dir+'critic')
        self.target_critic.save(self.chkpt_dir+'target_critic')

    def load_models(self):
        print('... loading models ...')
        self.actor = keras.models.load_model(self.chkpt_dir+'actor')
        self.target_actor = keras.models.load_model(self.chkpt_dir+'target_actor')
        self.critic = keras.models.load_model(self.chkpt_dir+'critic')
        self.target_critic = keras.models.load_model(self.chkpt_dir+'target_critic')

    def choose_action(self, observation, evaluate=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        if not evaluate:
            actions += tf.convert_to_tensor(self.noise(), dtype=tf.float32)
        # note that if the env has an action > 1, we have to multiply by
        # max action at some point
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)

        return actions[0]

    def learn(self):
        if len(self.memory.replay_buffer) < self.batch_size:
            return

        state, action, reward, state_, done = self.memory.mini_batch(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(state_, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        # Update critic network
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_)
            critic_value_ = tf.squeeze(self.target_critic(states_, target_actions), 1) # Take out the batch dimension
            critic_value = tf.squeeze(self.critic(states, actions), 1)

            target = rewards + self.gamma*critic_value_*(1 - np.array([done], dtype=np.int32))
            critic_loss = keras.losses.MSE(target, critic_value)

        params = self.critic.trainable_variables
        grads = tape.gradient(critic_loss, params)
        self.critic.optimizer.apply_gradients(zip(grads, params))

        # Update actor network
        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)
        
        params = self.actor.trainable_variables
        grads = tape.gradient(actor_loss, params)
        self.actor.optimizer.apply_gradients(zip(grads, params))

        self.update_network_parameters()