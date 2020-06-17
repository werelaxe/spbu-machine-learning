import numpy as np


class Buffer:
    def __init__(self, state_space_dim, action_space_dim, max_len=50000, batch_size=64):
        self.max_len = max_len
        self.batch_size = batch_size
        self.counter = 0
        self.states = np.zeros((self.max_len, state_space_dim))
        self.actions = np.zeros((self.max_len, action_space_dim))
        self.rewards = np.zeros((self.max_len, 1))
        self.next_states = np.zeros((self.max_len, state_space_dim))

    def remember(self, info):
        index = self.counter % self.max_len
        self.states[index] = info[0]
        self.actions[index] = info[1]
        self.next_states[index] = info[2]
        self.rewards[index] = info[3]
        self.counter += 1

    def sample(self):
        indices = np.random.choice(min(self.counter, self.max_len), self.batch_size)
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        return states, actions, next_states, rewards
