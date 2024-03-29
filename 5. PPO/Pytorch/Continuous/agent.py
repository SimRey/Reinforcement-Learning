import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta,Normal
import math


class BetaActor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, a_lr):
        super(BetaActor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net_width = net_width
        self.a_lr = a_lr

        self.fc1 = nn.Linear(self.state_dim, self.net_width)
        self.fc2 = nn.Linear(self.net_width, self.net_width)
        self.alpha = nn.Linear(self.net_width, self.action_dim)
        self.beta = nn.Linear(self.net_width, self.action_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.a_lr)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        
        alpha = F.softplus(self.alpha(x)) + 1.0
        beta = F.softplus(self.beta(x)) + 1.0
        
        return alpha, beta
    
    def get_dist(self,state):
        alpha, beta = self.forward(state)
        dist = Beta(alpha, beta)
        return dist
    
    def dist_mode(self,state):
        alpha, beta = self.forward(state)
        mode = (alpha) / (alpha + beta)
        return mode


class GaussianActor_musigma(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, a_lr):
        super(GaussianActor_musigma, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net_width = net_width
        self.a_lr = a_lr

        self.fc1 = nn.Linear(self.state_dim, self.net_width)
        self.fc2 = nn.Linear(self.net_width, self.net_width)
        self.mu = nn.Linear(self.net_width, self.action_dim)
        self.sigma = nn.Linear(self.net_width, self.action_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.a_lr)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        
        mu = torch.sigmoid(self.mu(x))
        sigma = F.softplus(self.sigma(x))
        
        return mu, sigma
    
    def get_dist(self,state):
        mu, sigma = self.forward(state)
        dist = Normal(mu, sigma)
        return dist


class GaussianActor_mu(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, a_lr, log_std=0):
        super(GaussianActor_mu, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net_width = net_width
        self.a_lr = a_lr

        self.fc1 = nn.Linear(self.state_dim, self.net_width)
        self.fc2 = nn.Linear(self.net_width, self.net_width)
        self.mu = nn.Linear(self.net_width, self.action_dim)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.0)
        
        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.a_lr)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        mu = torch.sigmoid(self.mu(a))
        return mu
    
    def get_dist(self,state):
        mu = self.forward(state)
        action_log_std = self.action_log_std.expand_as(mu)
        action_std = torch.exp(action_log_std)
        
        dist = Normal(mu, action_std)
        return dist


class Critic(nn.Module):
    def __init__(self, state_dim, net_width, c_lr):
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.net_width = net_width
        self.c_lr = c_lr
        
        self.fc1 = nn.Linear(self.state_dim, self.net_width)
        self.fc2 = nn.Linear(self.net_width, self.net_width)
        self.v = nn.Linear(self.net_width, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.c_lr)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        v = self.v(x)
        return v

class PPO(object):
    def __init__(self, state_dim, action_dim, env_with_Dead, gamma=0.99, gae_lambda=0.95,
        clip_rate=0.2, n_epochs=10, net_width=256, lr=3e-4, l2_reg = 1e-3,
        dist='Beta', optim_batch_size = 64, entropy_coef = 0, entropy_coef_decay = 0.9998):
        
        self.state_dim = state_dim
        self.env_with_Dead = env_with_Dead
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.dist = dist
        self.clip_rate = clip_rate
        self.n_epochs = n_epochs
        self.net_width = net_width
        self.lr = lr
        self.l2_reg = l2_reg
        self.optim_batch_size = optim_batch_size
        self.entropy_coef = entropy_coef
        self.entropy_coef_decay = entropy_coef_decay

        self.data = []

        if self.dist == 'Beta':
            self.actor = BetaActor(self.state_dim, self.action_dim, self.net_width, self.lr)
        elif self.dist == 'GS_ms':
            self.actor = GaussianActor_musigma(self.state_dim, self.action_dim, self.net_width, self.lr)
        elif self.dist == 'GS_m':
            self.actor = GaussianActor_mu(self.state_dim, self.action_dim, self.net_width, self.lr)
        else:
            print('Dist Error')
        
        self.critic = Critic(self.state_dim, self.net_width, self.lr)

    def select_action(self, state):#only used when interact with the env
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1))
            dist = self.actor.get_dist(state)
            a = dist.sample()
            
            if self.dist == 'Beta':
                a = torch.clamp(a, 0, 1)

            logprob_a = dist.log_prob(a).cpu().numpy().flatten()
            return a.cpu().numpy().flatten(), logprob_a
    
    
    def evaluate(self, state):#only used when evaluate the policy.Making the performance more stable
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1))
            if self.dist == 'Beta':
                a = self.actor.dist_mode(state)
            elif self.dist == 'GS_ms':
                a,b = self.actor(state)
            elif self.dist == 'GS_m':
                a = self.actor(state)
            return a.cpu().numpy().flatten(),0.0
    
    def train(self):
        self.entropy_coef*=self.entropy_coef_decay
        s, a, r, s_prime, logprob_a, dones, dws = self.make_batch()
    
    
        ''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_prime)
            
            '''dw for TD_target and Adv'''
            deltas = r + self.gamma * vs_ * (1 - dws) - vs
            
            deltas = deltas.cpu().flatten().numpy()
            adv = [0]
            
            '''done for GAE'''
            for dlt, done in zip(deltas[::-1], dones.cpu().flatten().numpy()[::-1]):
                advantage = dlt + self.gamma * self.gae_lambda * adv[-1] * (1 - done)
                adv.append(advantage)
            adv.reverse()
            adv = copy.deepcopy(adv[0:-1])
            adv = torch.tensor(adv).unsqueeze(1).float()
            td_target = adv + vs
            adv = (adv - adv.mean()) / ((adv.std()+1e-8))
        
        """Slice long trajectopy into short trajectory and perform mini-batch PPO update"""
        optim_iter_num = int(math.ceil(s.shape[0] / self.optim_batch_size))
        for i in range(self.n_epochs):
            
            #Shuffle the trajectory, Good for training
            perm = np.arange(s.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm)
            s, a, td_target, adv, logprob_a = s[perm].clone(), a[perm].clone(), \
                td_target[perm].clone(), adv[perm].clone(), logprob_a[perm].clone()
            
            '''update the actor'''
            for i in range(optim_iter_num):
                index = slice(i * self.optim_batch_size, min((i + 1) * self.optim_batch_size, s.shape[0]))
                distribution = self.actor.get_dist(s[index])
                dist_entropy = distribution.entropy().sum(1, keepdim=True)
                logprob_a_now = distribution.log_prob(a[index])
                ratio = torch.exp(logprob_a_now.sum(1,keepdim=True) - logprob_a[index].sum(1,keepdim=True))  # a/b == exp(log(a)-log(b))
                
                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
                a_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
                
                self.actor.optimizer.zero_grad()
                a_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor.optimizer.step()
            
            '''update the critic'''
            for i in range(optim_iter_num):
                index = slice(i * self.optim_batch_size, min((i + 1) * self.optim_batch_size, s.shape[0]))
                c_loss = (self.critic(s[index]) - td_target[index]).pow(2).mean()
                for name,param in self.critic.named_parameters():
                    if 'weight' in name:
                        c_loss += param.pow(2).sum() * self.l2_reg
                
                self.critic.optimizer.zero_grad()
                c_loss.backward()
                self.critic.optimizer.step()
    
    def make_batch(self):
        l = len(self.data)
        s_lst, a_lst, r_lst, s_prime_lst, logprob_a_lst, done_lst, dw_lst = \
            np.zeros((l,self.state_dim)), np.zeros((l,self.action_dim)), np.zeros((l,1)),\
                np.zeros((l,self.state_dim)), np.zeros((l,self.action_dim)), np.zeros((l,1)), np.zeros((l,1))
            
        for i,transition in enumerate(self.data):
            s_lst[i], a_lst[i], r_lst[i], s_prime_lst[i], logprob_a_lst[i], done_lst[i], dw_lst[i] = transition
        
        if not self.env_with_Dead:
            '''Important!!!'''
            # env_without_DeadAndWin: deltas = r + self.gamma * vs_ - vs
            # env_with_DeadAndWin: deltas = r + self.gamma * vs_ * (1 - dw) - vs
            dw_lst *=False

        self.data = [] #Clean history trajectory      
        
        '''list to tensor'''
        with torch.no_grad():
            s, a, r, s_prime, logprob_a, dones, dws = \
                torch.tensor(s_lst, dtype=torch.float), \
                torch.tensor(a_lst, dtype=torch.float), \
                torch.tensor(r_lst, dtype=torch.float), \
                torch.tensor(s_prime_lst, dtype=torch.float), \
                torch.tensor(logprob_a_lst, dtype=torch.float), \
                torch.tensor(done_lst, dtype=torch.float), \
                torch.tensor(dw_lst, dtype=torch.float),
        
        return s, a, r, s_prime, logprob_a, dones, dws
    
    
    def put_data(self, transition):
        self.data.append(transition)
    
    def save(self, episode):
        torch.save(self.critic.state_dict(), f"./model/ppo_critic{episode}.pth")
        torch.save(self.actor.state_dict(), f"./model/ppo_actor{episode}.pth")
    
    def best_save(self):
        torch.save(self.critic.state_dict(), f"./best_model/ppo_critic.pth")
        torch.save(self.actor.state_dict(), f"./best_model/ppo_actor.pth")
    
    def load(self,episode):
        self.critic.load_state_dict(torch.load(f"./model/ppo_critic{episode}.pth"))
        self.actor.load_state_dict(torch.load(f"./model/ppo_actor{episode}.pth"))
    
    def load_best(self):
        self.critic.load_state_dict(torch.load(f"./best_model/ppo_critic.pth"))
        self.actor.load_state_dict(torch.load(f"./best_model/ppo_actor.pth"))

