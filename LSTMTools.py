import torch
import torch.nn as nn
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_device
import torch.optim as optim

import gymnasium as gym
import numpy as np
from collections import deque

class FrameStackEnv(gym.Wrapper):
    """
    A simple wrapper that returns a stack of the last N observations 
    each time step.
    """
    def __init__(self, env, stack_size=30):
        super(FrameStackEnv, self).__init__(env)
        self.stack_size = stack_size
        self.frames = deque([], maxlen=stack_size)

        obs_shape = env.observation_space.shape  # e.g. (obs_dim,)
        # New observation space: (stack_size, obs_dim)
        self.observation_space = gym.spaces.Box(
            low=np.repeat(env.observation_space.low[None], stack_size, axis=0),
            high=np.repeat(env.observation_space.high[None], stack_size, axis=0),
            shape=(stack_size, *obs_shape),
            dtype=env.observation_space.dtype,
        )

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset()
        # Clear and stack the same observation for the initial state
        for _ in range(self.stack_size):
            self.frames.append(obs)
        return np.stack(self.frames, axis=0), {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return np.stack(self.frames, axis=0), reward, terminated, truncated, info



class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    A custom feature extractor that processes observations with an LSTM
    and outputs a fixed-sized feature vector.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim = 128):
        
        super().__init__(observation_space, features_dim)
        
        self.device = get_device("auto")
        obs_dim = observation_space.shape[1]

        hidden_size = 512  
        self.lstm = nn.LSTM(
            input_size=obs_dim, 
            hidden_size=hidden_size, 
            batch_first=True
        )
        
        # After the LSTM, we map from hidden_size -> features_dim
        self.linear1 = nn.Linear(hidden_size, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, features_dim)

        # Activations for between the linear layers
        self.activation = nn.Tanh()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations shape: (batch_size, 30, obs_dim), for example
        lstm_out, _ = self.lstm(observations)           # -> (batch_size, 30, lstm_hidden_size)
        last_timestep = lstm_out[:, -1, :]              # -> (batch_size, lstm_hidden_size)

        # Pass through 3 linear layers + Tanh
        x = self.activation(self.linear1(last_timestep))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))

        # Return final features to PPO
        return x

