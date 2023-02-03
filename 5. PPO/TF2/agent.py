import numpy as np
import tensorflow as tf 
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from memory import PPOMemory
from networks import ActorNetwork, CriticNetwork



class Agent:
    def __init__(self, n_actions, gamma=0.99, alpha=3e-4, gae_lambda=0.95, policy_clip=0.2, 
                 batch_size=64, n_epochs=10, chkpt_dir="models/"):
        self.gamma = gamma
        self.alpha = alpha
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.chkpt_dir = chkpt_dir
        
        self.actor = ActorNetwork(n_actions)
        self.critic = CriticNetwork()
        self.actor.compile(optimizer=Adam(learning_rate=self.alpha))
        self.critic.compile(optimizer=Adam(learning_rate=self.alpha))
        
        self.memory = PPOMemory(batch_size)

    def remember(self, state, state_, action, probs, reward, done):
        self.memory.store_memory(state, state_, action, probs, reward, done)

    def save_models(self):
        self.actor.save(self.chkpt_dir + "actor")
        self.critic.save(self.chkpt_dir + "critic")

    def load_models(self):
        self.actor = keras.models.load_model(self.chkpt_dir + "actor")
        self.critic = keras.models.load_model(self.chkpt_dir + "critic")

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        dist = self.actor(state)
        action = dist.sample()
        probs = dist.log_prob(action)

        action = action.numpy()[0]
        probs = probs.numpy()[0]

        return action, probs

    def calc_adv_and_returns(self, memories):
        states, new_states, r, dones = memories
        values = self.critic(states)
        values_ = self.critic(new_states)
        deltas = r + self.gamma * values_ - values
        deltas = deltas.numpy()

        adv = [0]
        for step in reversed(range(len(deltas))):
            advantage = deltas[step] + self.gamma * self.gae_lambda * adv[-1] * (1 - dones[step])
            adv.append(advantage)
        adv.reverse()
        adv = adv[:-1]
        adv = np.array(adv, dtype=np.float64)
        returns = adv + values
        adv = (adv - adv.mean()) / (adv.std()+1e-6)
        return adv, returns.numpy()

    
    def learn(self):
        state_arr, new_state_arr, action_arr, old_prob_arr, reward_arr, dones_arr = self.memory.recall()
        r = np.array(reward_arr.reshape((-1,1)), dtype=np.float64)
        adv, returns = self.calc_adv_and_returns((state_arr, new_state_arr, r, dones_arr))
        
        for epoch in range(self.n_epochs):
            batches = self.memory.generate_batches()
            for batch in batches:
                with tf.GradientTape(persistent=True) as tape:
                    states = tf.convert_to_tensor(state_arr[batch])
                    old_probs = tf.convert_to_tensor(old_prob_arr[batch])
                    actions = tf.convert_to_tensor(action_arr[batch])
                    dist = self.actor(states)
                    new_probs = dist.log_prob(actions)

                    prob_ratio = tf.math.exp(new_probs- old_probs)
                    weighted_probs = adv[batch] * prob_ratio
                    weighted_clipped_probs = tf.clip_by_value(
                        prob_ratio, 1-self.policy_clip, 1+self.policy_clip
                        ) * adv[batch]
                
                    actor_loss = -tf.math.minimum(weighted_probs, weighted_clipped_probs)
                    actor_loss = tf.math.reduce_mean(actor_loss)
                                        
                    critic_values = self.critic(state_arr[batch])
                    critic_loss = keras.losses.MSE(returns[batch], critic_values)
                    
                
                actor_params = self.actor.trainable_variables
                critic_params = self.critic.trainable_variables
                actor_grads = tape.gradient(actor_loss, actor_params)
                critic_grads = tape.gradient(critic_loss, critic_params)

                self.actor.optimizer.apply_gradients(zip(actor_grads, actor_params))
                self.critic.optimizer.apply_gradients(zip(critic_grads, critic_params))
                
        
        self.memory.clear_memory()