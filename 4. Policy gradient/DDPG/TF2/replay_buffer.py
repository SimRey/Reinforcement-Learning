import random
from collections import deque


class ReplayBuffer(object):
    def __init__(self, max_size):
        self.mem_size = max_size # Size of Replay Buffer
        self.replay_buffer = deque(maxlen=self.mem_size) # Creation of the Replay Buffer

    def store_transition(self, state, action, reward, state_, done):
        self.replay_buffer.append((state, action, reward, state_, done))
        
    def mini_batch(self, batch_size):
        batch = random.sample(self.replay_buffer, batch_size)
        zipped_batch = list(zip(*batch))
        states, actions, rewards, new_states, dones = zipped_batch 

        return states, actions, rewards, new_states, dones