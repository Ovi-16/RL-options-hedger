# replay_buffer.py
# Simple circular buffer for experience replay

import collections
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)
        self.state_dim = state_dim

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        s, a, r, s_next, d = map(np.stack, zip(*batch))
        return s, a, r, s_next, d

    def __len__(self):
        return len(self.buffer)