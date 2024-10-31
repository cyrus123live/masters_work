import StockData
import ModelTools
import TradingEnv
import math
import datetime as dt
import pandas as pd

parameters = {
    "starting_month": "2016-1",
    "ending_month": "2020-6",
    "train_months": 3,
    "test_months": 3,
    "trade_months": 3,
    "num_ppo": 1,
    "num_a2c": 1,
    "test_before_train": False,
    "training_rounds_per_contender": 1,
    "timesteps_between_check_PPO": 100000,
    "timesteps_between_check_A2C": 35000,
    "starting_cash": 1000000,
    "ent_coef": 0.1,
    "verbose": True,
    "buy_sell_action_space": "continuous", 
    'validation_parameter': "sharpe",
    "indicators": ["close_normalized", 'macd_normalized', 'rsi_normalized', 'cci_normalized', "adx_normalized"],
    "fees": 0, # Doesn't work yet
    # "tickers": ["spy"]
    # "tickers": ["spy", "eem", "fxi", "efa", "iev", "ewz", "efz", "fxi", "yxi", "iev", "epv", "ewz"]
    "tickers": ['AXP', 'AAPL', 'VZ', 'BA', 'CAT', 'JPM', 'CVX', 'KO', 'DIS', 'DD', 'XOM', 'HD', 'INTC', 'IBM', 'JNJ', 'MCD', 'MRK', 'MMM', 'NKE', 'PFE', 'PG', 'UNH', 'RTX', 'WMT', 'WBA', 'MSFT', 'CSCO', 'TRV', 'GS', 'V']
}

starting_month = dt.datetime(year=int(parameters["starting_month"].split("-")[0]), month=int(parameters["starting_month"].split("-")[1]), day=1)
ending_month = dt.datetime(year=int(parameters["ending_month"].split("-")[0]), month=int(parameters["ending_month"].split("-")[1]), day=1)
total_months = math.ceil((ending_month - starting_month).days / 30.44)

turbulence = StockData.calculate_turbulence(StockData.get_consecutive_months(starting_month - pd.DateOffset(months = parameters["train_months"] + parameters["test_months"]), total_months, parameters), parameters)

turbulence.to_csv("turbulence.csv")

print(turbulence)