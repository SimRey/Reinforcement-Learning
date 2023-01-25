import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from replay_buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork


class Agent:
    def __init__(self, n_actions,
        alpha=0.004, # Learning rate actor network
        beta=0.006, # Learning rate critic and value network (can be higher than actor network)
        env=None, gamma=0.99, mem_size=50000, tau=0.005, fc1_dims=400, fc2_dims=300, 
        batch_size=128, reward_scale=2, chkpt_dir='models/'):

        self.n_actions = n_actions
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mem_size = mem_size
        self.fname = chkpt_dir
        self.tau = tau
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.batch_size = batch_size
        self.chkpt_dir = chkpt_dir

        self.max_action = env.action_space.high[0]
        self.memory = ReplayBuffer(self.mem_size)        

        # Define the networks
        self.actor = ActorNetwork(n_actions=self.n_actions, fc1_dims=self.fc1_dims, 
                                  fc2_dims=self.fc2_dims, max_action=self.max_action)
        
        self.critic1 = CriticNetwork(fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims)
        self.critic2 = CriticNetwork(fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims)

        self.value = ValueNetwork(fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims)
        self.value_target = ValueNetwork(fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims)

        # Network compilation
        self.actor.compile(optimizer=Adam(learning_rate=self.alpha))
        self.critic1.compile(optimizer=Adam(learning_rate=self.beta))
        self.critic2.compile(optimizer=Adam(learning_rate=self.beta))
        self.value.compile(optimizer=Adam(learning_rate=self.beta))
        self.value_target.compile(optimizer=Adam(learning_rate=self.beta))

        self.scale = reward_scale
        self.update_network_parameters(tau=1) # Initialize to same values

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.value_target.weights
        for i, weight in enumerate(self.value.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.value_target.set_weights(weights)

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)


    def choose_action(self, observation, evaluate=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions, _ = self.actor.sample_normal(state)
        return actions[0]


    def learn(self):
        if len(self.memory.replay_buffer) < self.batch_size:
            return

        state, action, reward, state_, done = self.memory.mini_batch(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(state_, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        # Update value network
        with tf.GradientTape() as tape:
            value = self.value(states)
            value = tf.squeeze(value, 1)
            value_ = self.value_target(states_)
            value_ = tf.squeeze(value_, 1)

            current_policy_actions, log_probs = self.actor.sample_normal(states)
            log_probs = tf.squeeze(log_probs, 1)
        
            q1_new_pi = self.critic1(states, current_policy_actions)
            q2_new_pi = self.critic2(states, current_policy_actions)
            critic_value = tf.squeeze(tf.math.minimum(q1_new_pi, q2_new_pi), 1)

            value_target = critic_value - log_probs
            value_loss = 0.5 * keras.losses.MSE(value, value_target)
        params = self.value.trainable_variables
        grads = tape.gradient(value_loss, params)
        self.value.optimizer.apply_gradients(zip(grads, params))
    
        # Update actor network
        with tf.GradientTape() as tape:
            new_policy_actions, log_probs = self.actor.sample_normal(states)
            log_probs = tf.squeeze(log_probs, 1)
            q1_new_policy = self.critic1(states, new_policy_actions)
            q2_new_policy = self.critic2(states, new_policy_actions)
            critic_value = tf.squeeze(tf.math.minimum(
                                        q1_new_policy, q2_new_policy), 1)
            actor_loss = log_probs - critic_value
            actor_loss = tf.math.reduce_mean(actor_loss)
        params = self.actor.trainable_variables
        grads = tape.gradient(actor_loss, params)
        self.actor.optimizer.apply_gradients(zip(grads, params))

        with tf.GradientTape(persistent=True) as tape: # apply gradients twice for both networks
            value_ = tf.squeeze(self.value_target(states_), 1)
            q_hat = self.scale*rewards + self.gamma*value_*(1 - np.array([done], dtype=np.int32))
            q1_old_policy = tf.squeeze(self.critic1(states, actions), 1)
            q2_old_policy = tf.squeeze(self.critic2(states, actions), 1)
            critic1_loss = 0.5 * keras.losses.MSE(q1_old_policy, q_hat)
            critic2_loss = 0.5 * keras.losses.MSE(q2_old_policy, q_hat)
        params_1 = self.critic1.trainable_variables
        params_2 = self.critic2.trainable_variables
        grads_1 = tape.gradient(critic1_loss, params_1)
        grads_2 = tape.gradient(critic2_loss, params_2)

        self.critic1.optimizer.apply_gradients(zip(grads_1, params_1))
        self.critic2.optimizer.apply_gradients(zip(grads_2, params_2))

        self.update_network_parameters()


    def save_models(self):
        if len(self.memory.replay_buffer) > self.batch_size:
            print('... saving models ...')
            self.actor.save(self.fname+'actor')
            self.critic1.save(self.fname+'critic1')
            self.critic2.save(self.fname+'critic2')
            self.value.save(self.fname+'value')
            self.value_target.save(self.fname+'value_target')

    def load_models(self):
        print('... loading models ...')
        self.actor = keras.models.load_model(self.fname+'actor')
        self.critic1 = keras.models.load_model(self.fname+'critic1')
        self.critic2 = keras.models.load_model(self.fname+'critic2')
        self.value = keras.models.load_model(self.fname+'value')
        self.value_target = keras.models.load_model(self.fname+'value_target')