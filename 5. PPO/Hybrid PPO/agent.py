import torch as T
from memory import PPOMemory
from networks import HybridActorNetwork, HybridCriticNetwork


class Agent:
    def __init__(self, actions, input_dims, fc1_dims=128, fc2_dims=128, gamma=0.99, alpha=3e-4,
                 gae_lambda=0.95, policy_clip=0.2, batch_size=64, n_epochs=10):
        
        self.actions = actions
        self.input_dims = input_dims
        self.alpha = alpha
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        
        self.actor = HybridActorNetwork(self.actions, self.input_dims, 
            self.alpha, self.fc1_dims, self.fc2_dims)
        self.critic = HybridCriticNetwork(self.input_dims, self.alpha, 
            self.fc1_dims, self.fc2_dims)

        self.memory = PPOMemory(self.batch_size)

    def remember(self, state, state_, action_d, action_c, probs_d, probs_c, reward, done):
        self.memory.store_memory(state, state_, action_d, action_c, probs_d, probs_c, reward, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        with T.no_grad():
            state = T.tensor([observation], dtype=T.float)

            dist_d, dist_c = self.actor(state)

            # Discrete
            action_d = dist_d.sample()
            probs_d = T.squeeze(dist_d.log_prob(action_d)).item()
            action_d = T.squeeze(action_d).item()

            # Continuous
            action_c = dist_c.sample()
            probs_c = dist_c.log_prob(action_c)
            action_c = action_c.numpy().flatten()
            probs_c = probs_c.numpy().flatten()

            return action_d, probs_d, action_c, probs_c


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

    
    def learn(self):
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

                dist_d, _ = self.actor(states)
                new_probs = dist_d.log_prob(actions)
                prob_ratio = T.exp(new_probs - old_probs)
                weighted_probs = adv[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * adv[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs)

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

                _, dist_c = self.actor(states)
                new_probs = dist_c.log_prob(actions)
                prob_ratio = T.exp(new_probs.sum(1, keepdim=True) - old_probs.sum(1, keepdim=True))
                weighted_probs = adv[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * adv[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs)

                # Deactivate continuouse parameters
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
                critic_loss = (critic_value - returns[batch]).pow(2).mean()
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()
        
        self.memory.clear_memory()