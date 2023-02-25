import numpy as np

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        
        self.probs_d = []
        self.probs_c = []
        self.actions_d = []
        self.actions_c = []
        
        self.rewards = []
        self.dones = []
        self.new_states = []

        self.batch_size = batch_size

    def recall(self):
        return np.array(self.states), np.array(self.new_states), \
            np.array(self.actions_d), np.array(self.actions_c), \
               np.array(self.probs_d), np.array(self.probs_c), \
                np.array(self.rewards), np.array(self.dones)

    def generate_batches(self):
        n_states = len(self.states)
        # batch_start = np.arange(0, n_states, self.batch_size)
        n_batches = int(n_states // self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        # batches = [indices[i:i+self.batch_size] for i in batch_start]
        batches = [indices[i*self.batch_size:(i+1)*self.batch_size]
                   for i in range(n_batches)]
        return batches

    def store_memory(self, state, state_, action_d, action_c, probs_d, probs_c, reward, done):
        self.states.append(state)
        self.actions_d.append(action_d)
        self.actions_c.append(action_c)
        self.probs_d.append(probs_d)
        self.probs_c.append(probs_c)
        self.rewards.append(reward)
        self.dones.append(done)
        self.new_states.append(state_)

    def clear_memory(self):
       self.states = []
       self.probs_d = []
       self.probs_c = []
       self.actions_d = []
       self.actions_c = []
       self.rewards = []
       self.dones = []
       self.new_states = []