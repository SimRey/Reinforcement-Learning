import copy
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from replay_buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork
from noise import OUActionNoise
import math


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, fc1_dims, fc2_dims, a_lr):
        super(ActorNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.a_lr = a_lr

        self.fc1 = nn.Linear(*self.state_dim, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.a_lr)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = T.tanh(self.mu(x))

        return x

class Q_Critic(nn.Module):
    def __init__(self, state_dim, n_actions, fc1_dims, fc2_dims, c_lr):
        super(Q_Critic, self).__init__()

        self.state_dim = state_dim
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.c_lr = c_lr

        self.fc1 = nn.Linear(*self.state_dim + self.n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.c_lr)
    
    def forward(self, state, action):
        sa = T.cat([state, action], 1)
        sa = F.relu(self.fc1(sa))
        sa = F.relu(self.fc2(sa))
        sa = self.q(sa)
        
        return sa


class DDPG(object):
    def __init__(self, state_dim, n_actions,
        a_lr, # Learning rate actor network
        c_lr, # Learning rate critic network (can be higher than actor network)
        env, gamma, mem_size, tau, fc1_dims, fc2_dims, batch_size):

        self.state_dim = state_dim
        self.n_actions = n_actions
        self.a_lr = a_lr
        self.c_lr = c_lr
        self.gamma = gamma
        self.mem_size = mem_size
        self.tau = tau
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.batch_size = batch_size

        self.max_action = env.action_space.high[0]

        self.memory = ReplayBuffer(self.mem_size)
        self.noise = OUActionNoise(mu=np.zeros(self.n_actions,))

        self.actor = ActorNetwork(self.state_dim, self.n_actions, 
            self.fc1_dims, self.fc2_dims, self.a_lr)
        self.target_actor = ActorNetwork(self.state_dim, self.n_actions, 
            self.fc1_dims, self.fc2_dims, self.a_lr)
        self.critic = Q_Critic(self.state_dim, self.n_actions, 
            self.fc1_dims, self.fc2_dims, self.c_lr)
        self.target_critic = Q_Critic(self.state_dim, self.n_actions, 
            self.fc1_dims, self.fc2_dims, self.c_lr)

        self.update_network_parameters(tau=1) # Initialize to same values
    
    
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
             actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                 (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
    
    
    
    def select_action(self, state, evaluate=False):
        with T.no_grad():
            state = T.tensor([state], dtype=T.float)
            mu = self.actor.forward(state)
            if not evaluate:
                mu += T.tensor(self.noise(), dtype=T.float)
            mu = mu.clip(-1,1)*self.max_action
        
        return mu.numpy().flatten()


    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)
    
    def learn(self):
        if len(self.memory.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, states_, done = self.memory.mini_batch(self.batch_size)

        states = T.tensor(states, dtype=T.float)
        states_ = T.tensor(states_, dtype=T.float)
        actions = T.tensor(actions, dtype=T.float)
        rewards = T.tensor(rewards, dtype=T.float)
        done = T.tensor(done)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_actions = self.target_actor.forward(states_)
        critic_value_ = self.target_critic.forward(states_, target_actions)
        critic_value = self.critic.forward(states, actions)

        target = []
        for j in range(self.batch_size):
            targ = rewards[j] + self.gamma*critic_value_[j]*(1 - done[j])
            target.append(targ)
        target = T.tensor(target)
        target = target.view(self.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()


        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(states)
        self.actor.train()
        actor_loss = -self.critic.forward(states, mu)
        actor_loss = T.mean(actor_loss)

        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()
    
    
    def save(self,episode):
        T.save(self.critic.state_dict(), f"./model/ddpg_critic{episode}.pth")
        T.save(self.actor.state_dict(), f"./model/ddpg_actor{episode}.pth")
    
    def best_save(self):
        T.save(self.critic.state_dict(), f"./best_model/ddpg_critic.pth")
        T.save(self.actor.state_dict(), f"./best_model/ddpg_actor.pth")
    
    def load(self,episode):
        self.critic.load_state_dict(T.load(f"./model/ddpg_critic{episode}.pth"))
        self.actor.load_state_dict(T.load(f"./model/ddpg_actor{episode}.pth"))
    
    def load_best(self):
        self.critic.load_state_dict(T.load(f"./best_model/ddpg_critic.pth"))
        self.actor.load_state_dict(T.load(f"./best_model/ddpg_actor.pth"))