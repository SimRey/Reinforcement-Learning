import copy
from memory import PPOMemory
import numpy as np
import torch as T
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta,Normal
import math


class BetaActor(nn.Module):
    def __init__(self, input_dims, n_actions, net_width, a_lr):
        super(BetaActor, self).__init__()
        self.fc1 = nn.Linear(*input_dims, net_width)
        self.fc2 = nn.Linear(net_width, net_width)
        self.alpha = nn.Linear(net_width, n_actions)
        self.beta = nn.Linear(net_width, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=a_lr)
    
    def forward(self, state):
        x = T.tanh(self.fc1(state))
        x = T.tanh(self.fc2(x))
        alpha = F.softplus(self.alpha(x)) + 1.0
        beta = F.softplus(self.beta(x)) + 1.0
        
        return alpha, beta
    
    def get_dist(self,state):
        alpha,beta = self.forward(state)
        dist = Beta(alpha, beta)

        return dist
    
    def dist_mode(self,state):
        alpha, beta = self.forward(state)
        mode = (alpha) / (alpha + beta)

        return mode

class GaussianActor(nn.Module):
    def __init__(self, input_dims, n_actions, net_width, a_lr):
        super(GaussianActor, self).__init__()
        
        self.fc1 = nn.Linear(*input_dims, net_width)
        self.fc2 = nn.Linear(net_width, net_width)
        self.mu = nn.Linear(net_width, n_actions)
        self.sigma = nn.Linear(net_width, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=a_lr)
    
    def forward(self, state):
        x = T.tanh(self.fc1(state))
        x = T.tanh(self.fc2(x))
        mu = F.sigmoid(self.mu(x))
        sigma = F.softplus(self.sigma(x))
        
        return mu, sigma
    
    def get_dist(self, state):
        mu,sigma = self.forward(state)
        dist = Normal(mu,sigma)
        
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
        n_epochs=10, net_width=256, a_lr=3e-4, c_lr=3e-4, l2_reg=1e-3, dist='Beta',
        batch_size=64, entropy_coef=0, entropy_coef_decay=0.9998):

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
        self.dist = dist
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.entropy_coef_decay = entropy_coef_decay

        self.memory = PPOMemory(self.batch_size)

        
        if self.dist == 'Beta':
            self.actor = BetaActor(self.input_dims, self.n_actions, self.net_width, self.a_lr)
        elif self.dist == "GS":
            self.actor = GaussianActor(self.input_dims, self.n_actions, self.net_width, self.a_lr)
        else: print('Dist Error')

        self.critic = Critic(self.input_dims, self.net_width, self.c_lr)
    
    def select_action(self, state):#only used when interact with the env
        with T.no_grad():
            state = T.tensor([state], dtype=T.float)
            dist = self.actor.get_dist(state)
            action = T.clamp(dist.sample(), 0, 1)
            probs = dist.log_prob(action)

        return action.numpy().flatten(), probs.numpy().flatten()
    
    def remember(self, state, state_, action, probs, reward, done):
        self.memory.store_memory(state, state_, action, probs, reward, done)

    def evaluate(self, state):#only used when evaluate the policy.Making the performance more stable
        with T.no_grad():
            state = T.tensor([state], dtype=T.float)
            if self.dist == 'Beta':
                action = self.actor.dist_mode(state)
            elif self.dist == 'GS':
                action, _ = self.actor(state)
            else:
                print('Dist Error')

        return action.numpy().flatten(), 0.0
    
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
                dist_entropy = dist.entropy().sum(1, keepdim=True)
                new_probs = dist.log_prob(actions)
                prob_ratio = T.exp(new_probs.sum(1, keepdim=True) - old_probs.sum(1, keepdim=True))

                
                weighted_probs = adv[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * adv[batch]
                
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs) - self.entropy_coef * dist_entropy

                self.actor.optimizer.zero_grad()
                actor_loss.mean().backward()
                T.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor.optimizer.step()

                critic_value = self.critic(states)
                critic_loss = (critic_value - returns[batch]).pow(2).sum()* self.l2_reg
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