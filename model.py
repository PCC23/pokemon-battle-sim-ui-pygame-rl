# models.py
# This file contains:
#   1) QNet: a small neural network that outputs Q values for each action (move slot)
#   2) ReplayBuffer: a FIFO buffer of experience tuples for off policy RL training

import random  # used for uniform random sampling from the replay buffer
from collections import deque  # deque gives an efficient fixed size queue with maxlen eviction

import numpy as np  # used to pack sampled batches into arrays for training
import torch  # core PyTorch package
import torch.nn as nn  # neural network layers and base Module class


class QNet(nn.Module):
    """
    Simple MLP that maps an observation vector to 4 Q values, one per move slot.
    """

    def __init__(self, obs_size, n_actions=4, hidden=256):
        super().__init__()  # initialize the base nn.Module internals (parameters, buffers, etc.)

        # Build a feedforward network:
        # observation vector -> hidden layer -> hidden layer -> Q values (one per action)
        self.net = nn.Sequential(
            nn.Linear(int(obs_size), int(hidden)),  # first linear layer: obs_size features -> hidden units
            nn.ReLU(),  # nonlinearity so the network can learn non linear functions
            nn.Linear(int(hidden), int(hidden)),  # second linear layer: hidden units -> hidden units
            nn.ReLU(),  # another nonlinearity
            nn.Linear(int(hidden), int(n_actions)),  # output layer: hidden units -> Q value for each action
        )

    def forward(self, x):
        # Forward pass: given a batch of observations x, return predicted Q values per action
        return self.net(x)


class ReplayBuffer:
    """
    Stores transitions (s, a, r, s2, done) for off policy learning.
    Uniform sampling breaks temporal correlation.
    """

    def __init__(self, capacity=50000):
        # Create a deque that automatically discards the oldest element when it exceeds capacity
        self.buf = deque(maxlen=int(capacity))

    def add(self, s, a, r, s2, done):
        # Store one transition tuple:
        #   s: current state / observation
        #   a: action taken (cast to int to avoid weird types)
        #   r: reward received (cast to float for numeric stability)
        #   s2: next state / observation
        #   done: episode termination flag (cast to float so it can be used in math: 0.0 or 1.0)
        self.buf.append((s, int(a), float(r), s2, float(done)))

    def sample(self, batch_size):
        # Sample a random minibatch of transitions (uniform random sampling)
        batch = random.sample(self.buf, int(batch_size))

        # Unzip the list of tuples into five separate tuples:
        #   s: states, a: actions, r: rewards, s2: next states, d: done flags
        s, a, r, s2, d = zip(*batch)

        # Return the batch in numpy arrays with training friendly dtypes:
        #   states and next states are stacked into (batch_size, obs_size)
        #   actions are int64 (typical for indexing/gather in PyTorch)
        #   rewards and done flags are float32
        return (
            np.stack(s, axis=0).astype(np.float32),  # batch of states
            np.array(a, dtype=np.int64),  # batch of actions
            np.array(r, dtype=np.float32),  # batch of rewards
            np.stack(s2, axis=0).astype(np.float32),  # batch of next states
            np.array(d, dtype=np.float32),  # batch of done flags (0.0 or 1.0)
        )

    def __len__(self):
        # Allow len(buffer) to return the current number of stored transitions
        return len(self.buf)
