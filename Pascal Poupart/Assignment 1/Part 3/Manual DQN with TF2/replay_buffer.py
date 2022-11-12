import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_size, input_dims):
        self.mem_size = max_size       # Memory size of the Replay Buffer
        self.mem_cntr = 0              # Memory counter

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32) # This terminal memory is used to avoid counting an extra reward when the process has finished

    
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size  # Once the replay buffer is full, the index restarts as zero, otherwise is the index

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)   # In this case True = 0, False = 1, so terminal memory has to be encoded in this way

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False) # repalce = False --> taken out of the pool

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        action = self.action_memory[batch]
        reward = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, states_, action, reward, terminal