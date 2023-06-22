import torch as T
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np
from replay_buffer import ReplayBuffer



class Network(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(Network, self).__init__()

        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        
        # Building the neural Network
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)  
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions


class DQN(object):
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

        self.model = Network(self.lr, self.input_dims, 128, 128, self.n_actions)
        self.target = Network(self.lr, self.input_dims, 128, 128, self.n_actions)

        self.criterion = nn.MSELoss()


    def store_transition(self, state, action, reward, state_, done):
        """The following funtion is used to 'remember' the previously stated function in the replay buffer file"""

        self.memory.store_transition(state, action, reward, state_, done)


    def choose_action(self, observation):
        """This function performs an epsilon greedy action choice"""

        if np.random.random() > self.epsilon:
            state = T.tensor([observation],dtype=T.float)
            actions = self.model.forward(state)
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
        if len(self.memory.replay_buffer) < self.batch_size:
            return

        self.model.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.memory.mini_batch(self.batch_size)
        indices = np.arange(self.batch_size)

        # Predict targets for all states from the sample
        targets = self.target.forward(states)
        q_values = self.model.forward(states)
        
        # Predict Q-Values for all new states from the sample
        q_next = T.max(self.target.forward(states_), dim = 1).values

        # Replace the targets values with the according function
        targets[indices, actions] = rewards + self.gamma * q_next*(1 - dones)

        loss = self.criterion(targets, q_values)
        loss.backward()
        self.model.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
    
    
    def save(self,episode):
        T.save(self.model.state_dict(), f"./model/dqn{episode}.pth")
    
    def best_save(self):
        T.save(self.model.state_dict(), f"./best_model/dqn.pth")
    
    def load(self,episode):
        self.model.load_state_dict(T.load(f"./model/dqn{episode}.pth"))
    
    def load_best(self):
        self.model.load_state_dict(T.load(f"./best_model/dqn.pth"))
