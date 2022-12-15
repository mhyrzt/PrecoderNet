import random
import numpy as np
from collections import deque


class ReplyBuffer:
    def __init__(self, max_len: int, batch_size: int):
        self.actions = deque(maxlen=max_len)
        self.rewards = deque(maxlen=max_len)
        self.next_states = deque(maxlen=max_len)
        self.current_states = deque(maxlen=max_len)

        self.maxlen = max_len
        self.batch_size = batch_size

    def add(self, current_state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray):
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.current_states.append(current_state)
        return self

    def _get_ids(self):
        n = range(len(self.current_states))
        return random.sample(n, self.batch_size)

    def sample(self):
        ids = self._get_ids()

        actions = self.actions[ids]
        rewards = self.rewards[ids]
        next_states = self.next_states[ids]
        current_states = self.current_states[ids]

        return (current_states, actions, rewards, next_states)

    def __len__(self):
        return len(self.current_states)

    def is_filled(self):
        return len(self) >= self.maxlen

    def can_sample(self):
        return len(self) >= self.batch_size
