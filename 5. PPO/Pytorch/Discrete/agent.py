import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import copy
import math


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, a_lr):
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net_width = net_width
        self.a_lr = a_lr

        self.fc1 = nn.Linear(self.state_dim, self.net_width)
        self.fc2 = nn.Linear(self.net_width, self.net_width)
        self.pi = nn.Linear(self.net_width, self.action_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.a_lr)


    def forward(self, state, dim=0):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))

        logits = self.pi(x)
        prob = F.softmax(logits, dim=dim)

        return prob



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

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.v(x)
        return x



class PPO(object):
    def __init__(self, env_with_Dead, state_dim, action_dim, gamma=0.99, gae_lambda=0.95,
            net_width=200, lr=1e-4, policy_clip=0.2, n_epochs=10, batch_size=64,
            l2_reg=1e-3, entropy_coef=1e-3, adv_normalization=True,
            entropy_coef_decay = 0.99):

        self.env_with_Dead = env_with_Dead
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.net_width = net_width
        self.lr = lr
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.optim_batch_size = batch_size

        self.l2_reg = l2_reg
        self.entropy_coef = entropy_coef
        self.adv_normalization = adv_normalization
        self.entropy_coef_decay = entropy_coef_decay

        self.actor = Actor(self.s_dim, self.a_dim, self.net_width, self.lr)
        self.critic = Critic(self.s_dim, self.net_width, self.lr)
        
        # Replay buffer
        self.data = []
        


    def select_action(self, state):
        '''Stochastic Policy'''
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float)
            pi = self.actor(state)
            dist = Categorical(pi)
            action = dist.sample().item()
            pi_a = pi[action].item()
        return action, pi_a


    def evaluate(self, state):
        '''Deterministic Policy'''
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float)
            pi = self.actor(state)
            a = torch.argmax(pi).item()
        return a,1.0


    def train(self):
        s, a, r, s_prime, old_prob_a, dones, dws = self.make_batch()
        self.entropy_coef *= self.entropy_coef_decay #exploring decay

        ''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_prime)

            '''dw(dead and win) for TD_target and Adv'''
            deltas = r + self.gamma*vs_ * (1 - dws) - vs
            deltas = deltas.cpu().flatten().numpy()
            adv = [0]

            '''done for GAE'''
            for dlt, done in zip(deltas[::-1], dones.cpu().flatten().numpy()[::-1]):
                advantage = dlt + self.gamma * self.gae_lambda * adv[-1] * (1 - done)
                adv.append(advantage)
            adv.reverse()
            adv = copy.deepcopy(adv[:-1])
            adv = torch.tensor(adv).unsqueeze(1).float()
            td_target = adv + vs
            if self.adv_normalization:
                adv = (adv - adv.mean()) / ((adv.std() + 1e-8))  

        """PPO update"""
        #Slice long trajectopy into short trajectory and perform mini-batch PPO update
        optim_iter_num = int(math.ceil(s.shape[0] / self.optim_batch_size))

        for _ in range(self.n_epochs):
            #Shuffle the trajectory, Good for training
            perm = np.arange(s.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm)
            s, a, td_target, adv, old_prob_a = \
                s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), old_prob_a[perm].clone()
            '''mini-batch PPO update'''
            for i in range(optim_iter_num):
                index = slice(i * self.optim_batch_size, min((i + 1) * self.optim_batch_size, s.shape[0]))

                '''actor update'''
                prob = self.actor(s[index], dim=1)
                entropy = Categorical(prob).entropy().sum(0, keepdim=True)
                prob_a = prob.gather(1, a[index])
                ratio = torch.exp(torch.log(prob_a) - torch.log(old_prob_a[index]))


                surr1 = -ratio * adv[index]
                surr2 = -torch.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * adv[index]
                a_loss = torch.max(surr1, surr2) - self.entropy_coef * entropy

                self.actor.optimizer.zero_grad()
                a_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor.optimizer.step()

                '''critic update'''
                c_loss = F.mse_loss(td_target[index], self.critic(s[index]))
                for name, param in self.critic.named_parameters():
                    if 'weight' in name:
                        c_loss += param.pow(2).sum() * self.l2_reg

                self.critic.optimizer.zero_grad()
                c_loss.backward()
                self.critic.optimizer.step()
        return a_loss, c_loss, entropy

    def make_batch(self):
        l = len(self.data)
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst, dw_lst = \
            np.zeros((l,self.s_dim)), np.zeros((l,1)), np.zeros((l,1)), np.zeros((l,self.s_dim)),\
                 np.zeros((l,1)), np.zeros((l,1)), np.zeros((l,1))
            
        for i,transition in enumerate(self.data):
            s_lst[i], a_lst[i],r_lst[i] ,s_prime_lst[i] ,prob_a_lst[i] ,done_lst[i] ,dw_lst[i] = transition
        
        if not self.env_with_Dead:
            '''Important!!!'''
            # env_without_DeadAndWin: deltas = r + self.gamma * vs_ - vs
            # env_with_DeadAndWin: deltas = r + self.gamma * vs_ * (1 - dw) - vs
            dw_lst *=False

        self.data = [] #Clean history trajectory

        '''list to tensor'''
        with torch.no_grad():
            s,a,r,s_prime,prob_a,dones, dws = \
                torch.tensor(s_lst, dtype=torch.float), \
                torch.tensor(a_lst, dtype=torch.int64), \
                torch.tensor(r_lst, dtype=torch.float), \
                torch.tensor(s_prime_lst, dtype=torch.float), \
                torch.tensor(prob_a_lst, dtype=torch.float), \
                torch.tensor(done_lst, dtype=torch.float), \
                torch.tensor(dw_lst, dtype=torch.float)

        return s, a, r, s_prime, prob_a,dones,dws

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