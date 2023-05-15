import os
import numpy as np
import copy
import math
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from memory import PPOMemory


class Actor(nn.Module):
    def __init__(self, input_dims, n_actions, net_width, a_lr):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(*input_dims, net_width)
        self.fc2 = nn.Linear(net_width, net_width)
        self.pi = nn.Linear(net_width, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=a_lr)

    def forward(self, state):
        x = T.tanh(self.fc1(state))
        x = T.tanh(self.fc2(x))
        pi = F.softmax(self.pi(x), dim=1)
        return pi

    
    def get_dist(self,state):
        pi = self.forward(state)
        dist = Categorical(pi)
        return dist
        

class Critic(nn.Module):
    def __init__(self, input_dims, net_width, c_lr):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(*input_dims, net_width)
        self.fc2 = nn.Linear(net_width, net_width)
        self.v = nn.Linear(net_width, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=c_lr)

    def forward(self, state):
        x = T.tanh(self.fc1(state))
        x = T.tanh(self.fc2(x))
        v = self.v(x)
        return v



class PPO(object):
    def __init__(self, input_dims, n_actions, gamma=0.99, gae_lambda=0.95, policy_clip=0.2, 
            n_epochs=10, net_width=256, a_lr=3e-4, c_lr=3e-4, batch_size=64, l2_reg=1e-3,
            entropy_coef = 1e-3, entropy_coef_decay = 0.99):

        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.net_width = net_width
        self.a_lr = a_lr
        self.c_lr = c_lr
        self.l2_reg = l2_reg        
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.entropy_coef_decay = entropy_coef_decay

        self.memory = PPOMemory(self.batch_size)

        self.actor = Actor(self.input_dims, self.n_actions, self.net_width, self.a_lr)
        self.critic = Critic(self.input_dims, self.net_width, self.c_lr)

    
    def remember(self, state, state_, action, probs, reward, done):
        self.memory.store_memory(state, state_, action, probs, reward, done)  


    def choose_action(self, observation):
        '''Stochastic Policy'''
        with T.no_grad():
            state = T.tensor([observation], dtype=T.float)
            dist = self.actor.get_dist(state)
            action = dist.sample()
            probs = dist.log_prob(action)
            probs = T.squeeze(dist.log_prob(action)).item()
            action = T.squeeze(action).item()

        return action, probs

    def evaluate(self, observation):
        '''Deterministic Policy'''
        with T.no_grad():
            state = T.tensor([observation], dtype=T.float)
            pi = self.actor.forward(state)
            a = T.argmax(pi).item()
        
        return a, 1.0
    
    def calc_adv_and_returns(self, memories):
        states, new_states, r, dones = memories
        with T.no_grad():
            values = self.critic(states)
            values_ = self.critic(new_states)
            deltas = r + self.gamma * values_ - values
            deltas = deltas.flatten().numpy()
            adv = [0]
            for step in reversed(range(deltas.shape[0])):
                advantage = deltas[step] + self.gamma * self.gae_lambda * adv[-1] * (1 - dones[step])
                adv.append(advantage)
            adv.reverse()
            adv = adv[:-1]
            adv = T.tensor(adv).float().unsqueeze(1)
            returns = adv + values
            adv = (adv - adv.mean()) / (adv.std()+1e-6)
        return adv, returns


    def train(self):
        self.entropy_coef*=self.entropy_coef_decay
        state_arr, new_state_arr, action_arr, old_prob_arr, reward_arr, dones_arr = self.memory.recall()

        state_arr, action_arr, old_prob_arr, new_state_arr, r = T.tensor(state_arr, dtype=T.float), \
            T.tensor(action_arr, dtype=T.float), T.tensor(old_prob_arr, dtype=T.float), \
                T.tensor(new_state_arr, dtype=T.float), T.tensor(reward_arr, dtype=T.float).unsqueeze(1)

        adv, returns = self.calc_adv_and_returns((state_arr, new_state_arr, r, dones_arr))  

        for epoch in range(self.n_epochs):
            batches = self.memory.generate_batches()
            for batch in batches:
                states = state_arr[batch]
                old_probs = old_prob_arr[batch]
                actions = action_arr[batch]

                dist = self.actor.get_dist(states)
                entropy = dist.entropy().sum(0, keepdim=True)
                new_probs = dist.log_prob(actions)
                prob_ratio = T.exp(new_probs - old_probs)
                weighted_probs = adv[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * adv[batch]
                
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs) - self.entropy_coef * entropy

                self.actor.optimizer.zero_grad()
                actor_loss.mean().backward()
                T.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor.optimizer.step()

                critic_value = self.critic(states)
                critic_loss = (critic_value - returns[batch]).pow(2).mean()* self.l2_reg
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()
        
        self.memory.clear_memory()


    def save(self,episode):
        T.save(self.critic.state_dict(), f"./model/ppo_critic{episode}.pth")
        T.save(self.actor.state_dict(), f"./model/ppo_actor{episode}.pth")
    
    def best_save(self):
        T.save(self.critic.state_dict(), f"./best_model/ppo_critic.pth")
        T.save(self.actor.state_dict(), f"./best_model/ppo_actor.pth")
    
    def load(self,episode):
        self.critic.load_state_dict(T.load(f"./model/ppo_critic{episode}.pth"))
        self.actor.load_state_dict(T.load(f"./model/ppo_actor{episode}.pth"))
    
    def load_best(self):
        self.critic.load_state_dict(T.load(f"./best_model/ppo_critic.pth"))
        self.actor.load_state_dict(T.load(f"./best_model/ppo_actor.pth"))