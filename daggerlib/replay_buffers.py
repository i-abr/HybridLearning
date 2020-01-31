import random
import numpy as np
from itertools import compress


class ReplayBuffer(object):


    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0


    def push(self, state, action, nextstate, expert_action):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, nextstate, expert_action)
        self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, nextstate, expert_action = map(np.stack, zip(*batch))
        return state, action, nextstate, expert_action

    def __len__(self):
        return len(self.buffer)
