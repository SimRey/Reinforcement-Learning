import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta, Categorical


class HybridActorNetwork(nn.Module):
    def __init__(self, actions, input_dims, lr,
                 fc1_dims, fc2_dims, chkpt_dir='models/'):
        super(HybridActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir,'hppo')
        self.input_dims = input_dims
        self.lr = lr
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        
        self.actions = actions
        # Discrete actions
        self.d_actions = self.actions["discrete"].n
        # Continuous actions
        self.c_actions = self.actions["continuous"].shape[0]

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # Discrete actions
        self.pi = nn.Linear(self.fc2_dims, self.d_actions)
        # Continuous actions
        self.alpha = nn.Linear(self.fc2_dims, self.c_actions)
        self.beta = nn.Linear(self.fc2_dims, self.c_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
       

    def forward(self, state):
        x = T.tanh(self.fc1(state))
        x = T.tanh(self.fc2(x))
        
        # Discrete distribution
        pi = F.softmax(self.pi(x), dim=1)
        dist_d = Categorical(pi)

        # Continuous distribution
        alpha = F.relu(self.alpha(x)) + 1.0
        beta = F.relu(self.beta(x)) + 1.0
        dist_c = Beta(alpha, beta)

        return dist_d, dist_c

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))



class HybridCriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha,
                 fc1_dims, fc2_dims, chkpt_dir='models/'):
        super(HybridCriticNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir,'critic_continuous_ppo')
        self.input_dims = input_dims
        self.alpha = alpha
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)

    def forward(self, state):
        x = T.tanh(self.fc1(state))
        x = T.tanh(self.fc2(x))
        v = self.v(x)

        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))