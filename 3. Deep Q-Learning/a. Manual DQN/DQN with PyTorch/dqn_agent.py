import torch as T
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np
from replay_buffer import ReplayBuffer



class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()

        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        
        # Building the neural Network
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)  

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions


class Agent(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, mem_size, batch_size, 
                 eps_min=0.01, eps_dec=5e-7, replace=100):
    
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]

        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size)

        self.model = DeepQNetwork(self.lr, self.input_dims, 256, 256, self.n_actions)
        self.target = DeepQNetwork(self.lr, self.input_dims, 256, 256, self.n_actions)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)


    def store_transition(self, state, action, reward, state_, done):
        """The following funtion is used to 'remember' the previously stated function in the replay buffer file"""

        self.memory.store_transition(state, action, reward, state_, done)


    def choose_action(self, observation):
        """This function performs an epsilon greedy action choice"""

        if np.random.random() > self.epsilon:
            state = T.tensor([observation],dtype=T.float)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

       
    def replace_target_network(self):
        """Funtion used to replace the w-values in the target neural network with the w-values of the 
        model neural network, this replacement occurs every n steps"""

        if self.learn_step_counter > 0 and self.learn_step_counter % self.replace_target_cnt == 0:
            self.target.load_state_dict(self.model.state_dict())


    def decrement_epsilon(self):
        """Funtion used to compute the decrement of epsilon"""
        if self.epsilon > self.eps_min:
            self.epsilon -= self.eps_dec
        else:
            self.epsilon = self.eps_min


    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.model.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.memory.mini_batch(self.batch_size)
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next

        loss = self.q_eval.loss(q_target, q_pred)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()