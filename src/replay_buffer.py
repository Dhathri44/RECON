import random
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=50000, seed=42):
        self.memory = deque(maxlen=capacity)
        self.rng = random.Random(seed)
    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = self.rng.sample(self.memory, batch_size)
        s,a,r,ns,d = zip(*batch)
        return (np.asarray(s,dtype=np.float32), np.asarray(a,dtype=np.int64),
                np.asarray(r,dtype=np.float32), np.asarray(ns,dtype=np.float32),
                np.asarray(d,dtype=np.float32))
    def __len__(self):
        return len(self.memory)
