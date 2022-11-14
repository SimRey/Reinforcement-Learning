import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from network import DeepQNetwork
from replay_buffer import ReplayBuffer


class Agent:
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = DeepQNetwork(input_dims, n_actions, 256, 256)
        self.q_eval.compile(optimizer=Adam(learning_rate=lr))
        self.q_next = DeepQNetwork(input_dims, n_actions, 256, 256)
        self.q_next.compile(optimizer=Adam(learning_rate=lr))

    

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)


    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = tf.convert_to_tensor([observation])
            actions = self.q_eval(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]
        else:
            action = np.random.choice(self.action_space)
        return action

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.set_weights(self.q_eval.get_weights())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)

        indices = tf.range(self.batch_size, dtype=tf.int32)
        action_indices = tf.stack([indices, actions], axis=1)

        with tf.GradientTape() as tape:
            q_pred = tf.gather_nd(self.q_eval(states), indices=action_indices)
            q_next = self.q_next(states_)

            max_actions = tf.math.argmax(q_next, axis=1, output_type=tf.int32)
            max_action_idx = tf.stack([indices, max_actions], axis=1)

            q_target = rewards + \
                self.gamma*tf.gather_nd(q_next, indices=max_action_idx) *\
                (1 - dones.numpy())

            loss = keras.losses.MSE(q_pred, q_target)

        params = self.q_eval.trainable_variables
        grads = tape.gradient(loss, params)

        self.q_eval.optimizer.apply_gradients(zip(grads, params))

        self.learn_step_counter += 1
        self.decrement_epsilon()