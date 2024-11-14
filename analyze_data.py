import StockData
import datetime as dt
import pandas as pd
import ModelTools
import matplotlib.pyplot as plt

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
    "tickers": ["BTCUSDT", "ETHUSDT"]
    # "tickers": ["btc"]
    # "tickers": ["spy", "eem", "fxi", "efa", "iev", "ewz", "efz", "fxi", "yxi", "iev", "epv", "ewz"]
    # "tickers": ['AXP', 'AAPL', 'VZ', 'BA', 'CAT', 'JPM', 'CVX', 'KO', 'DIS', 'DD', 'XOM', 'HD', 'INTC', 'IBM', 'JNJ', 'MCD', 'MRK', 'MMM', 'NKE', 'PFE', 'PG', 'UNH', 'RTX', 'WMT', 'WBA', 'MSFT', 'CSCO', 'TRV', 'GS', 'V']
}

# for i in range(54):
#     date = dt.datetime(year=2020, month=1, day=1) + pd.DateOffset(months=i)
#     print(date)
#     data = StockData.get_consecutive_months(date, 1, parameters)   
#     combined_index = pd.concat(data).index.unique()
#     missing_minutes = {}
#     for j, df in enumerate(data):
#         missing = combined_index.difference(df.index)
#         # missing_minutes[f'DF{j}'] = missing
#         print(missing)
#         for t in parameters["tickers"]:
#             print(t)
#             new_data = pd.read_csv(f"stock_data/{t}_data/{date.strftime('%Y-%m')}.csv")
#             for missing_time in missing:
#                 if missing_time in new_data.index:
#                     print(new_data.loc[missing_time])


data = StockData.get_consecutive_months(dt.datetime(year=2022, month=1, day=1), 24, parameters)
# Create a combined set of unique timestamps
# missing_indexes = []
# for df in data:
#     missing_indexes.append(combined_index.difference(df.index))

# for df in data:
#     df.drop(index=missing_indexes)

common_index = data[0].index
for df in data[1:]:
    common_index = common_index.intersection(df.index)

data = [df.loc[common_index] for df in data]
combined_index = pd.concat(data).index.unique()

# combined_index = pd.date_range(start=data[0].index[0], end=data[0].index[-1], freq='min')

# Find missing timestamps for each DataFrame
missing_minutes = {}
for i, df in enumerate(data, start=1):
    missing = combined_index.difference(df.index)
    missing_minutes[f'DF{i}'] = missing

# Print missing timestamps for each DataFrame

print([d.shape[0] for d in data])

figure = plt.figure()
p = figure.add_subplot()
i = 0

to_plot = pd.DataFrame(index=data[0].index)
colours = ModelTools.get_distinct_colors(len(data))
for i, (df_name, missing) in enumerate(missing_minutes.items()):
    print(f"{df_name} is missing data at the following minutes:")
    print(missing)
    print()

    [p.axvline(x = j, color = colours[i]) for j in missing if not (j.hour == 0 and j.minute <= 40)]
for i in range(len(data)):
    close_data = pd.DataFrame(index=data[i].index)
    close_data["close"] = [float(data[i]['close'].iloc[j]) for j in range(len(data[i]))]
    to_plot[f'close_{i}'] = close_data["close"] / float(data[i]["close"].iloc[0])
    p.plot(to_plot[f'close_{i}'], color=colours[i])

p.legend()
plt.show()