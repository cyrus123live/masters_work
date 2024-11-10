import StockData
import datetime as dt
import pandas as pd

parameters = {
    "starting_month": "2021-1",
    "ending_month": "2024-6",
    "train_months": 1,
    "test_months": 1,
    "trade_months": 1,
    "num_ppo": 16,
    "num_a2c": 16,
    "test_before_train": False,
    "training_rounds_per_contender": 1,
    # "timesteps_between_check_PPO": 100000,
    # "timesteps_between_check_A2C": 35000,
    # "timesteps_between_check_PPO": 10000, # From ensemble ipynb
    # "timesteps_between_check_A2C": 10000, # From ensemble ipynb
    "timesteps_between_check_PPO": 50000, 
    "timesteps_between_check_A2C": 50000, 
    "starting_cash": 1000000,
    "verbose": True,
    "buy_sell_action_space": "discrete", 
    'validation_parameter': "sharpe",
    "indicators": ["close_normalized", 'macd_normalized', 'rsi_normalized', 'cci_normalized', "adx_normalized"],
    "fees": 0, # Doesn't work yet
    "turbulence_threshold": 140, # From original ensemble code
    "use_turbulence": False,
    "t": "minutely",
    # "turbulence_threshold": 201.71875, # From ensemble ipynb
    # "turbulence_threshold": 1,
    "tickers": ["btc", "ETHUSDT"]
    # "tickers": ["spy", "eem", "fxi", "efa", "iev", "ewz", "efz", "fxi", "yxi", "iev", "epv", "ewz"]
    # "tickers": ['AXP', 'AAPL', 'VZ', 'BA', 'CAT', 'JPM', 'CVX', 'KO', 'DIS', 'DD', 'XOM', 'HD', 'INTC', 'IBM', 'JNJ', 'MCD', 'MRK', 'MMM', 'NKE', 'PFE', 'PG', 'UNH', 'RTX', 'WMT', 'WBA', 'MSFT', 'CSCO', 'TRV', 'GS', 'V']
}

data = StockData.get_consecutive_months(dt.datetime(year=2020, month=1, day=1), 54, parameters)
# Create a combined set of unique timestamps
combined_index = pd.concat(data).index.unique()

# Find missing timestamps for each DataFrame
missing_minutes = {}
for i, df in enumerate(data, start=1):
    missing = combined_index.difference(df.index)
    missing_minutes[f'DF{i}'] = missing

# Print missing timestamps for each DataFrame
for df_name, missing in missing_minutes.items():
    print(f"{df_name} is missing data at the following minutes:")
    print(missing)
    print()

print([d.shape[0] for d in data])