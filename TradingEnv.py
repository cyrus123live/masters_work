import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import time
import math

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data, parameters, starting_cash, turbulence, trading = False):
        super(TradingEnv, self).__init__()

        self.data = data
        self.reward_range = (-np.inf, np.inf)

        self.starting_cash = starting_cash
        self.parameters = parameters
        self.k = [int(self.starting_cash / df['close'].iloc[0]) for df in data]
        
        # Define action and observation spaces
        if parameters["buy_sell_action_space"] == "continuous":
            self.action_space = spaces.Box(low=-1, high=1, shape=(len(data),), dtype=np.float64)
        else:
            self.action_space = spaces.Discrete(n=len(data) + 1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(data) * (len(self.parameters['indicators']) + 1) + 1,), dtype=np.float64
        ) 
        
        # Set starting point
        self.current_step = 0
        self.max_steps = len(self.data[0]) - 1
        
        # Initialize portfolio
        self.cash = self.starting_cash
        self.stock = [0 for i in data]
        self.total_value = self.cash
        self.last_action = []
        self.turbulence = turbulence
        self.trading = trading

        self.c_selling = parameters["fees"]
        self.c_buying = parameters["fees"] # 0.5% fees is 0.0005

    def _get_obs(self):

        observations = []
        for i, df in enumerate(self.data):
            observations.extend(df[self.parameters['indicators']].iloc[self.current_step].to_list())
            observations.append(self.stock[i] / self.k[i])
        observations.append(self.cash / self.starting_cash)
        return observations

    def _take_action(self, action):

        self.last_action = action

        # if self.trading:
        #     print("stock:", self.stock)
        #     print("cash:", self.cash)
            # print("action:", action)

        if self.trading:
            turbulence = self.turbulence[self.turbulence["datadate"] == self.data[0].index[self.current_step]]["turbulence"].iloc[0]
            if turbulence >= self.parameters["turbulence_threshold"]:
                portfolio_value = sum([df["close"].iloc[self.current_step] * self.stock[i] for i, df in enumerate(self.data)]) + self.cash
                self.cash = portfolio_value
                self.stock = [0 for i in range(len(self.data))]
                print(f"Turbulence activated on date: {self.data[0].index[self.current_step]}")
                return

        if self.parameters["buy_sell_action_space"] == "discrete":

            decision = int(action)
            portfolio_value = sum([df["close"].iloc[self.current_step] * self.stock[i] for i, df in enumerate(self.data)]) + self.cash
            self.stock = [0 for i in range(len(self.data))]
            self.cash = 0
            if decision > 0:
                self.stock[decision - 1] = portfolio_value / self.data[decision - 1]["close"].iloc[self.current_step]
            else:
                self.cash = portfolio_value

        else:

            for i, df in enumerate(self.data):

                # Naive solution from ensemble: go through list and update based on buy and sell decisions, not taking into account that money may be ran out by the time we get to last stocks in list
                if action[i] < 0 and self.stock[i] > 0:
                    to_sell = min(self.stock[i], abs(action[i]) * self.k[i])
                    self.cash += to_sell * df["close"].iloc[self.current_step] * (1 - self.c_selling)
                    self.stock[i] -= to_sell
                    # print(f"sold {to_sell} of stock {i}, now have {self.cash} cash and {self.stock[i]} stock")
                
                # Buy if action is positive and cash is available
                elif action[i] > 0 and self.cash > 0:
                    max_stock = self.cash / df["close"].iloc[self.current_step]
                    to_buy = min(max_stock, action[i] * self.k[i])
                    self.cash -= to_buy * df["close"].iloc[self.current_step] * (1 + self.c_buying)
                    self.stock[i] += to_buy
                    # print(f"bought {to_buy} of stock {i}, now have {self.cash} cash and {self.stock[i]} stock")

        # print("resulting stock:", self.stock)
        # print("resulting cash:", self.cash)
        # time.sleep(5)

        # Fix rounding errors
        if self.cash < 0:
            self.cash = 0

    def step(self, action):
        self._take_action(action)
        
        # Move to the next time step
        self.current_step += 1
        
        # Calculate portfolio value at the new step
        new_total_value = sum([df["close"].iloc[self.current_step] * self.stock[i] for i, df in enumerate(self.data)]) + self.cash       

        # Reward scaling from ensemble
        reward = (new_total_value - self.total_value) * 1e-4
        
        self.total_value = new_total_value

        # Define whether the episode is finished
        terminated = self.current_step >= self.max_steps
        truncated = False

        # Get next observation
        obs = self._get_obs()
        info = {'total_value': self.total_value}

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):

        self.cash = self.starting_cash
        self.stock = [0 for i in self.data]
        self.total_value = self.cash
        self.current_step = 0
        self.last_action = []

        return self._get_obs(), {}

    def render(self, mode='human', close=False):
        if mode == 'human':
            output = {"portfolio_value": self.total_value, "cash": self.cash, "shares": self.stock, "closes": [float(df["close"].iloc[self.current_step]) for df in self.data], "last action": self.last_action}
            # print("From env: " + str(output) + "\n")
            return output