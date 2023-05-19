import copy
from memory import PPOMemory
import numpy as np
import torch as T
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Categorical
import math

class HybridActorNetwork(nn.Module):
    def __init__(self, input_dims, actions, net_width, a_lr):
        super(HybridActorNetwork, self).__init__()

        self.input_dims = input_dims
        self.a_lr = a_lr
        self.net_width = net_width

        self.actions = actions
        # Discrete actions
        self.d_actions = self.actions["discrete"].n
        # Continuous actions
        self.c_actions = self.actions["continuous"].shape[0]

        self.fc1 = nn.Linear(*self.input_dims, self.net_width)
        self.fc2 = nn.Linear(self.net_width, self.net_width)
        # Discrete actions
        self.pi = nn.Linear(self.net_width, self.d_actions)
        # Continuous actions
        self.alpha = nn.Linear(self.net_width, self.c_actions)
        self.beta = nn.Linear(self.net_width, self.c_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.a_lr)
       

    def forward(self, state):
        x = T.tanh(self.fc1(state))
        x = T.tanh(self.fc2(x))

        # Discrete
        pi = F.softmax(self.pi(x), dim=1)

        # Continuous
        alpha = F.softplus(self.alpha(x)) + 1.0
        beta = F.softplus(self.beta(x)) + 1.0

        return pi, [alpha, beta]
    
    def get_dist(self,state):
        pi, ab = self.forward(state)

        # Discrete distribution
        dist_d = Categorical(pi)

        # Continuous distribution
        dist_c = Beta(*ab)
        return dist_d, dist_c
    
    def dist_mode(self,state):
        # Only for contiuous part
        _, ab = self.forward(state)
        alpha, beta = ab
        mode = (alpha) / (alpha + beta)
        return mode


class HybridCriticNetwork(nn.Module):
    def __init__(self, input_dims, net_width, c_lr):
        super(HybridCriticNetwork, self).__init__()

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
    def __init__(self, input_dims, actions, gamma=0.99, gae_lambda=0.95, policy_clip=0.2, 
        n_epochs=10, net_width=256, a_lr=3e-4, c_lr=3e-4, l2_reg=1e-3, batch_size=64, 
        entropy_coef=0, entropy_coef_decay=0.9998):

        self.input_dims = input_dims
        self.actions = actions
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

        self.actor = HybridActorNetwork(self.input_dims, self.actions, self.net_width, self.a_lr) 
        self.critic = HybridCriticNetwork(self.input_dims, self.net_width, self.c_lr)


    
    def select_action(self, state):#only used when interact with the env
        with T.no_grad():
            state = T.tensor([state], dtype=T.float)
            dist_d, dist_c = self.actor.get_dist(state)

            # Discrete
            action_d = dist_d.sample()
            probs_d = T.squeeze(dist_d.log_prob(action_d)).item()
            action_d = T.squeeze(action_d).item()

            # Continuous
            action_c = T.clamp(dist_c.sample(), 0, 1)
            probs_c = dist_c.log_prob(action_c)
            action_c = action_c.numpy().flatten()
            probs_c = probs_c.numpy().flatten()

        return action_d, probs_d, action_c, probs_c

    
    def remember(self, state, state_, action_d, action_c, probs_d, probs_c, reward, done):
        self.memory.store_memory(state, state_, action_d, action_c, probs_d, probs_c, reward, done)


    def evaluate(self, state):#only used when evaluate the policy. Making the performance more stable
        with T.no_grad():
            state = T.tensor([state], dtype=T.float)
            
            pi,_ = self.actor.forward(state)
            a_d = T.argmax(pi).item()

            a_c = self.actor.dist_mode(state)
            a_c = a_c.numpy().flatten()
        
        return a_d, a_c

    
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

        state_arr, new_state_arr, actions_d_arr, actions_c_arr, \
                    old_prob_d_arr, old_prob_c_arr, reward_arr, dones_arr = self.memory.recall()

        state_arr = T.tensor(state_arr, dtype=T.float)
        new_state_arr = T.tensor(new_state_arr, dtype=T.float)
        actions_d_arr = T.tensor(actions_d_arr, dtype=T.float)
        actions_c_arr = T.tensor(actions_c_arr, dtype=T.float)        
        old_prob_d_arr = T.tensor(old_prob_d_arr, dtype=T.float)
        old_prob_c_arr = T.tensor(old_prob_c_arr, dtype=T.float)
        r = T.tensor(reward_arr, dtype=T.float).unsqueeze(1)

        adv, returns = self.calc_adv_and_returns((state_arr, new_state_arr, r, dones_arr))


        for epoch in range(self.n_epochs):
            batches = self.memory.generate_batches()
            
            for batch in batches:
                states = state_arr[batch]

                # Discrete update
                old_probs = old_prob_d_arr[batch]
                actions = actions_d_arr[batch]
                dist_d, _ = self.actor.get_dist(states)
                entropy = dist_d.entropy().sum(0, keepdim=True)
                new_probs = dist_d.log_prob(actions)
                prob_ratio = T.exp(new_probs - old_probs)
                
                weighted_probs = adv[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * adv[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs) - self.entropy_coef * entropy

                # Deactivate continuouse parameters
                total_params = sum([1 for param in self.actor.parameters()])-1
                params_c = []
                for j in range(4):
                    val = total_params - j
                    params_c.append(val)
                
                for i, param in enumerate(self.actor.parameters()):
                    if i in params_c:
                        param.requires_grad = False

                self.actor.optimizer.zero_grad()
                actor_loss.mean().backward()
                self.actor.optimizer.step()

                # Activate all parameters again
                for param in self.actor.parameters():
                    param.requires_grad = True

                
                # Continuous update
                old_probs = old_prob_c_arr[batch]
                actions = actions_c_arr[batch]

                _, dist_c = self.actor.get_dist(states)
                entropy = dist_c.entropy().sum(1, keepdim=True)
                new_probs = dist_c.log_prob(actions)
                prob_ratio = T.exp(new_probs.sum(1, keepdim=True) - old_probs.sum(1, keepdim=True))
                weighted_probs = adv[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * adv[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs) - self.entropy_coef * entropy

                # Deactivate discrete parameters
                total_params = sum([1 for param in self.actor.parameters()])-1
                params_d = []
                for j in range(6):
                    if j >= 4:
                        val = total_params - j
                        params_d.append(val)
                
                for i, param in enumerate(self.actor.parameters()):
                    if i in params_d:
                        param.requires_grad = False

                self.actor.optimizer.zero_grad()
                actor_loss.mean().backward()
                self.actor.optimizer.step()

                # Activate all parameters again
                for param in self.actor.parameters():
                    param.requires_grad = True

        
                # Critic network update        
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