import torch as T
import numpy as np
import torch.nn.functional as F
from replay_buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork
from noise import OUActionNoise

class Agent:
    def __init__(self, alpha, beta, input_dims, tau, env, gamma, n_actions, 
                 mem_size=50000, fc1_dims=400, fc2_dims=300, batch_size=64):
        
        self.alpha = alpha
        self.beta = beta
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.gamma = gamma
        self.mem_size = mem_size
        self.tau = tau
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.batch_size = batch_size

        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        self.memory = ReplayBuffer(self.mem_size)
        self.noise = OUActionNoise(mu=np.zeros(self.n_actions,))

        self.actor = ActorNetwork(alpha=self.alpha, input_dims=self.input_dims, 
                                  fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims, 
                                  n_actions=self.n_actions, name="Actor")
            
        self.critic = CriticNetwork(beta=self.beta, input_dims=self.input_dims, 
                                  fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims, 
                                  n_actions=self.n_actions, name="Critic")
        
        self.target_actor = ActorNetwork(alpha=self.alpha, input_dims=self.input_dims, 
                                  fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims, 
                                  n_actions=self.n_actions, name="TargetActor")

        self.target_critic = CriticNetwork(beta=self.beta, input_dims=self.input_dims, 
                                  fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims, 
                                  n_actions=self.n_actions, name="TargetCritic")
        
        self.update_network_parameters(tau=1) # Initialize to same values

  
   
    def choose_action(self, observation):
        self.actor.eval() # eval don't want to calculate statistics
        state = T.tensor([observation], dtype=T.float)
        mu = self.actor.forward(state)
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float)
        self.actor.train() # after eval important to state
        
        return mu_prime.detach().numpy()[0]

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

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
       
                