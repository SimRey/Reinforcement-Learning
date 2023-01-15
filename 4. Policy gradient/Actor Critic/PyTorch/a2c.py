import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1_dims, fc2_dims):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)
        self.v = nn.Linear(fc2_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = F.softmax(self.pi(x), dim=-1)
        v = self.v(x)

        return (pi, v)

class Agent():
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, 
                 gamma=0.99):
        self.gamma = gamma
        self.lr = lr
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.actor_critic = ActorCriticNetwork(lr, input_dims, n_actions, 
                                               fc1_dims, fc2_dims)
        self.log_probs = None

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float)
        probabilities, _ = self.actor_critic.forward(state)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.log_probs = log_probs

        return action.item()

    def learn(self, state, reward, state_, done, n):
        self.actor_critic.optimizer.zero_grad()

        state = T.tensor([state], dtype=T.float)
        state_ = T.tensor([state_], dtype=T.float)
        reward = T.tensor(reward, dtype=T.float)

        probabilities, Qw = self.actor_critic.forward(state)
        _, Qw_dash = self.actor_critic.forward(state_)
        Qw = T.squeeze(Qw)
        Qw_dash = T.squeeze(Qw_dash)
        probabilities = T.squeeze(probabilities)

        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        
        delta = reward + self.gamma*T.max(Qw_dash)*(1-int(done)) - Qw[action]
        advantage = reward + self.gamma*T.max(Qw_dash)*(1-int(done)) - sum([p*v for p,v in zip(probabilities, Qw)])

        actor_loss = -(self.log_probs*advantage)*self.gamma**n
        critic_loss = (delta**2)*self.gamma**n

        (actor_loss + critic_loss).backward()
        self.actor_critic.optimizer.step()
