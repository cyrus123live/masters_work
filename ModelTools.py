from stable_baselines3 import PPO
from TradingEnv import TradingEnv
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import StockData
import logging
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
import json
import csv
import math

def read_history_from_file(name):
    return pd.read_csv("history/" + name + ".csv", parse_dates=['timestamp'], index_col='timestamp')

def write_history_to_file(history, name="test"):
    history.to_csv("history/" + name + ".csv")

def get_stats_from_history(history):
    trading_days_per_year = 252 # estimate
    trading_minutes_per_year = trading_days_per_year * 540 # 540 minutes from 7 am to 4pm
    number_of_years = (history.index[-1] - history.index[0]).days / trading_days_per_year

    history['returns'] = np.log(history['portfolio_value'] / history['portfolio_value'].shift(1))
    mean_return = history['returns'].mean()
    std_return = history['returns'].std()

    cumulative_return = history.iloc[-1]['portfolio_value'] / history.iloc[0]['portfolio_value'] - 1
    annual_return = math.pow(1 + cumulative_return, (1/number_of_years)) - 1
    annual_volatility = std_return * np.sqrt(trading_minutes_per_year)
    sharpe_ratio = (mean_return / std_return) * np.sqrt(trading_minutes_per_year)

    rolling_max = history['portfolio_value'].cummax()
    drawdown = (history['portfolio_value'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    print(f"Cumulative return: {cumulative_return * 100:.2f}%")
    print(f"Annual return: {annual_volatility * 100:.2f}%")
    print(f"Annual volatility: {annual_return * 100:.2f}%")
    print(f"Sharpe ratio: {sharpe_ratio}")
    print(f"Max drawdown: {max_drawdown * 100:.2f}%")

def plot_history(history):
    to_plot = pd.DataFrame(index=history.index)
    to_plot['close'] = history["close"] / history.iloc[0]["close"]
    to_plot['portfolio'] = history["portfolio_value"] / history.iloc[0]["portfolio_value"]

    figure = plt.figure()
    p = figure.add_subplot()

    p.plot(to_plot['close'], label="Stock Movement")
    p.plot(to_plot['portfolio'], label="Portfolio Value")
    p.legend()

    plt.show()

# Returns a history dataframe
def test_model(model, test_data, starting_cash = 10000000):

    history = []
    k = starting_cash / test_data.iloc[0]["Close"]
    cash = starting_cash
    held = 0
    for i in range(test_data.shape[0]):

        data = test_data.iloc[i]
        # obs = np.array(test_data[test_data.filter(regex='_Scaled$').columns].iloc[i].tolist() + [np.clip(2 * held / k - 1, -1, 1), np.clip(2 * cash / starting_cash - 1, -1, 1)])
        # obs = np.array(test_data[["Close_Normalized", "Change_Normalized", "D_HL_Normalized"]].iloc[i].tolist() + [held / k, cash / starting_cash])
        obs = np.array(test_data[["Close_Normalized", "MACD_Normalized", "RSI_Normalized", "CCI_Normalized", "ADX_Normalized"]].iloc[i].tolist() + [held / k, cash / starting_cash])
        
        action = model.predict(obs, deterministic=True)[0][0]

        if action < 0:
            cash += held * data["Close"]
            held = 0
        else:
            to_buy = min(cash / data["Close"], action * k)
            cash -= to_buy * data["Close"]
            held += to_buy

        history.append({"portfolio_value": cash + held * data["Close"], "close": data["Close"], "cash": cash, "held": held})

    return pd.DataFrame(history, index=test_data.index)


def train_PPO(train_data, test_data, training_rounds_per_contender):

    train_env = Monitor(TradingEnv(train_data))
    model = PPO("MlpPolicy", train_env, verbose=0)
    best_model = model
    best_score = 0
    
    for i in range(training_rounds_per_contender):

        model.learn(total_timesteps=train_data.shape[0] - 1, progress_bar=False, reset_num_timesteps=False)
        score = test_model(model, test_data).iloc[-1]["portfolio_value"] # For now measuring score of model with just ending portfolio value, TODO: make this sharpe or sortino
        if score > best_score:
            best_model = model
            best_score = score

        print(f"- Ended training round {i + 1}/{training_rounds_per_contender} with score {score:.2f}")

    return {"model": best_model, "score": best_score}
