import numpy as np
import tensorflow as tf 
import tensorflow.keras as keras
from tensorflow.keras.optimizer import Adam
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
        self.actor.complie(optimizer=Adam(learning_rate=self.alpha))
        self.critic.complie(optimizer=Adam(learning_rate=self.alpha))
        
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
        print(values, values.shape)
        values_ = self.critic(new_states)
        deltas = r + self.gamma * values_ - values
        deltas = deltas.numpy()[0]
        adv = [0]
        for step in reversed(range(deltas.shape[0])):
            advantage = deltas[step] + self.gamma * self.gae_lambda * adv[-1] * (1 - dones[step])
            adv.append(advantage)
        adv.reverse()
        adv = adv[:-1]
        adv = np.array([adv], dtype=np.float64)
        print('adv', adv, adv.shape)
        returns = adv + values.numpy()
        print('returns', returns, returns.shape)
        adv = (adv - adv.mean()) / (adv.std()+1e-6)
        return adv, returns

    
    def learn(self):
        state_arr, new_state_arr, action_arr, old_prob_arr, reward_arr, dones_arr = self.memory.recall()
        r = np.array([reward_arr], dtype=np.float64)
        adv, returns = self.calc_adv_and_returns((state_arr, new_state_arr, r, dones_arr))
        
        for epoch in range(self.n_epochs):
            batches = self.memory.generate_batches()
            for batch in batches:
                states = state_arr[batch]
                old_probs = old_prob_arr[batch]
                actions = action_arr[batch]

                dist = self.actor(states)
                new_probs = dist.log_prob(actions)
                prob_ratio = T.exp(new_probs.sum(1, keepdim=True) - old_probs.sum(1, keepdim=True))
                weighted_probs = adv[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * adv[batch]
                
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs)

                self.actor.optimizer.zero_grad()
                actor_loss.mean().backward()
                self.actor.optimizer.step()

                critic_value = self.critic(states)
                critic_loss = (critic_value - returns[batch]).pow(2).mean()
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()
        
        self.memory.clear_memory()