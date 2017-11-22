
import random
import numpy as np
import tensorflow as tf
from npienv import *
from f import *
from mcst import *


class Memory:
    """
    This class provides an abstraction to store the [s, a, r, a'] elements of each iteration.
    Instead of using tuples (as other implementations do), the information is stored in lists
    that get returned as another list of dictionaries with each key corresponding to either
    "state", "action", "reward", "nextState" or "isFinal".
    """

    def __init__(self, size):
        self.size = size
        self.currentPosition = 0
        self.states = []
        self.pis = []
        self.rewards = []
        self.newStates = []
        self.finals = []

    def getMiniBatch(self, size):
        indices = random.sample(np.arange(len(self.states)), min(size, len(self.states)))
        miniBatch = []
        for index in indices:
            miniBatch.append({'state': self.states[index], 'pi': self.pis[index], 'reward': self.rewards[index]})
        return miniBatch

    def getCurrentSize(self):
        return len(self.states)

    def getMemory(self, index):
        return {'state': self.states[index], 'action': self.actions[index], 'reward': self.rewards[index],
                'newState': self.newStates[index], 'isFinal': self.finals[index]}

    def addMemory(self, state, pi, reward):
        if (self.currentPosition >= self.size - 1):
            self.currentPosition = 0
        if (len(self.states) > self.size):
            self.states[self.currentPosition] = state
            self.pis[self.currentPosition] = pi
            self.rewards[self.currentPosition] = reward
        else:
            self.states.append(state)
            self.pis.append(pi)
            self.rewards.append(reward)

        self.currentPosition += 1
