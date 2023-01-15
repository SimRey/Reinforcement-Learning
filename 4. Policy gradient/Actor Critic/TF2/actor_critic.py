import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
import numpy as np


class ActorCritic(keras.Model):
    def __init__(self, n_actions, fc1_dims, fc2_dims):
        super(ActorCritic, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.pi = Dense(n_actions, activation='softmax')
        self.v = Dense(1, activation=None)

    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(value)
        pi = self.pi(value)

        v = self.v(value)

        return (pi, v)
    


class Agent():
    def __init__(self, lr, gamma, n_actions, fc1_dims, fc2_dims):
        self.gamma = gamma
        self.lr = lr
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.action = 0

        self.actor_critic = ActorCritic(self.n_actions, self.fc1_dims, self.fc2_dims)
        self.actor_critic.compile(optimizer=Adam(learning_rate=self.lr))


    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        probs,_ = self.actor_critic(state)
        action_probs = tfp.distributions.Categorical(probs=probs)
        action = action_probs.sample()
        self.action = action

        return action.numpy()[0]


    def learn(self, state, state_, reward, done, n):

        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        done = int(done)

        with tf.GradientTape() as tape:
            probs, critic_value = self.actor_critic(state)
            _, critic_value_ = self.actor_critic(state_)
            critic_value = tf.squeeze(critic_value)
            critic_value_ = tf.squeeze(critic_value_)

            action_probs = tfp.distributions.Categorical(probs=probs)
            log_probs = action_probs.log_prob(self.action)
            

            delta = reward + self.gamma*critic_value_*(1-done) - critic_value

            actor_loss = -(log_probs*delta)*self.gamma**n
            critic_loss = (delta**2*self.gamma**n)
            loss = actor_loss + critic_loss

        params = self.actor_critic.trainable_variables
        grads = tape.gradient(loss, params)
        self.actor_critic.optimizer.apply_gradients(zip(grads, params))