from stable_baselines3 import PPO
from stable_baselines3 import A2C
from TradingEnv import TradingEnv
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import datetime as dt
import StockData
import logging
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
import json
import csv
import math
import random
import numpy as np
import torch
import os
import colorsys
import re
import pickle
import time
import copy
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan


class Logger():
    def __init__(self, run_folder_name):
        self.run_folder_name = run_folder_name

    def print_out(self, s):
        print(s)
        with open(f"{self.run_folder_name}/run.log", 'a') as f:
            f.write(str(s) + "\n")
        

def print_parameters(run_folder_name):
    print("\nrun " + run_folder_name)
    try:
        print("\nParameters:\n")
        with open(f"{run_folder_name}/parameters.json", 'r') as f:
            # print(json.dumps(json.loads(f.read()), indent=1))
            data = json.loads(f.read())
        for d in data:
            print(f"{d}: {data[d]}")
        print("\n")
    except Exception as e:
        print("Parameters File not found")


def combine_trade_window_histories(run_folder_name):

    combined_history = pd.DataFrame()

    folders = [d for d in os.listdir(f"{run_folder_name}") if "20" in d]
    folders.sort(key=lambda x: (int(x.split("-")[0]), int(x.split("-")[1])))

    for f in folders:
        try:
            combined_history = pd.concat([combined_history, read_history_from_file(f"{run_folder_name}/{f}/trade_window_history")])
        except:
            continue

    write_history_to_file(combined_history, f"{run_folder_name}/run_history")
    return combined_history

def make_dir(name):
    os.makedirs(name, exist_ok=True)
    os.chmod(name, 0o755)

def read_history_from_file(name):
    return pd.read_csv(name + ".csv", parse_dates=['date'], index_col='date')

def write_history_to_file(history, name="test"):
    history.to_csv(name + ".csv")


def get_cumulative_and_annual_returns(history, col):

    num_years = (history.index[-1].year - history.index[0].year + ((history.index[-1].month + 1) / 12) - (history.index[0].month / 12))

    cumulative_return = history.iloc[-1][col] / history.iloc[0][col] - 1
    annual_return = math.pow(1 + cumulative_return, (1/num_years)) - 1

    return cumulative_return, annual_return


def get_sharpe_and_volatility(history, col):

    # Plus one to last month because it ends at last day
    # num_years = (history.index[-1] - history.index[0]).days / 365.25 # by the day
    num_years = (history.index[-1].year - history.index[0].year + ((history.index[-1].month + 1) / 12) - (history.index[0].month / 12))
    trading_minutes_per_year = history.shape[0] / num_years

    history = history
    history['returns'] = history[col].pct_change(1)
    mean_return = history['returns'].mean()
    std_return = history['returns'].std()
    if std_return == 0: std_return = 0.0001

    history.dropna(inplace=True)

    sharpe = (mean_return / std_return) * np.sqrt(trading_minutes_per_year)
    volatility = std_return * np.sqrt(trading_minutes_per_year)

    return sharpe, volatility


def get_max_drawdown(history, col):

    rolling_max = history[col].cummax()
    drawdown = (history[col] - rolling_max) / rolling_max
    return drawdown.min()


def get_buy_hold_strategy(history, parameters):
    equal_parts_shares = [(parameters["starting_cash"] / len(parameters["tickers"])) / history.iloc[0]["closes"][i] for i in range(len(parameters["tickers"]))]
    buy_hold_history = copy.deepcopy(history)
    buy_hold_history["portfolio_value"] = [sum([history.iloc[j]["closes"][i] * equal_parts_shares[i] for i in range(len(parameters["tickers"]))]) for j in range(len(history))]  
    return buy_hold_history


def print_stats_from_history(history, parameters):

    print(f"\nRun Statistics for run [{history.index[0]}, {history.index[-1]}]:\n")

    buy_hold_history = get_buy_hold_strategy(history, parameters)
    # for i, ticker in enumerate(parameters["tickers"]):
    #     ticker_history = pd.DataFrame(index=history["closes"].index)
    #     ticker_history["close"] = [float(history["closes"].iloc[j][i]) for j in range(len(history["closes"]))]
    
    #     sharpe, volatility = get_sharpe_and_volatility(ticker_history, 'close')
    #     cumulative, annual = get_cumulative_and_annual_returns(ticker_history, 'close')
    #     max_drawdown = get_max_drawdown(ticker_history, 'close')

    #     print(f"\n\n{ticker} Buy and Hold Strategy:\n")
    #     print(f"- Cumulative return: {cumulative * 100:.2f}%")
    #     print(f"- Annual return: {annual * 100:.2f}%")
    #     print(f"- Annual volatility: {volatility * 100:.2f}%")
    #     print(f"- Sharpe ratio: {sharpe:.2f}")
    #     print(f"- Max drawdown: {max_drawdown * 100:.2f}%")

    sharpe, volatility = get_sharpe_and_volatility(buy_hold_history, 'portfolio_value')
    cumulative, annual = get_cumulative_and_annual_returns(buy_hold_history, 'portfolio_value')
    max_drawdown = get_max_drawdown(buy_hold_history, 'portfolio_value')

    print(f"\nBuy and Hold Strategy:\n")
    print(f"- Cumulative return: {cumulative * 100:.2f}%")
    print(f"- Annual return: {annual * 100:.2f}%")
    print(f"- Annual volatility: {volatility * 100:.2f}%")
    print(f"- Sharpe ratio: {sharpe:.2f}")
    print(f"- Max drawdown: {max_drawdown * 100:.2f}%")
    
    sharpe, volatility = get_sharpe_and_volatility(history, 'portfolio_value')
    cumulative, annual = get_cumulative_and_annual_returns(history, 'portfolio_value')
    max_drawdown = get_max_drawdown(history, 'portfolio_value')

    print(f"\n\nTest Strategy:\n")
    print(f"- Cumulative return: {cumulative * 100:.2f}%")
    print(f"- Annual return: {annual * 100:.2f}%")
    print(f"- Annual volatility: {volatility * 100:.2f}%")
    print(f"- Sharpe ratio: {sharpe:.2f}")
    print(f"- Max drawdown: {max_drawdown * 100:.2f}%")

    print("\n")

def get_distinct_colors(n):
    hues = [i / n for i in range(n)]
    random.shuffle(hues)  # Shuffle to add randomness
    colors = [colorsys.hsv_to_rgb(hue, 0.1, 0.9) for hue in hues]
    return colors

def plot_history(history, parameters):

    figure = plt.figure()
    p = figure.add_subplot()
    i = 0

    to_plot = pd.DataFrame(index=history.index)
    if "closes" in history.columns:
        colours = get_distinct_colors(len(history["closes"].iloc[0]))
        for i in range(len(history["closes"].iloc[0])):
            close_data = pd.DataFrame(index=history["closes"].index)
            close_data["close"] = [float(history["closes"].iloc[j][i]) for j in range(len(history["closes"]))]
            to_plot[f'close_{i}'] = [float(c) for c in close_data["close"] / float(history["closes"].iloc[0][i])]
            p.plot(to_plot[f'close_{i}'], color=colours[i])
    else:
        to_plot['close'] = history["close"] / history.iloc[0]["close"]
        p.plot(to_plot['close'], label="Stock Movement")
    
    buy_hold_history = get_buy_hold_strategy(history, parameters)

    to_plot['buy_hold'] = buy_hold_history["portfolio_value"] / buy_hold_history.iloc[0]["portfolio_value"]
    to_plot['portfolio'] = history["portfolio_value"] / history.iloc[0]["portfolio_value"]
    p.plot(to_plot['portfolio'], label="Portfolio Value")
    p.plot(to_plot['buy_hold'], label="Buy and Hold Strategy", color="black")
    # [p.axvline(x = i, color = 'b') for i in pd.date_range(history.index[0], history.index[-1], freq='QS')]
    p.legend()
    plt.show()


# Returns a history dataframe using TradingEnv
def test_model(model, test_data, parameters, cash, turbulence, trading = False):

    test_env = Monitor(TradingEnv(test_data, parameters, cash, turbulence, trading))
    obs, info = test_env.reset()
    history = [copy.deepcopy(test_env.render())]
    # model.set_env(test_env)
    for i in range(test_data[0].shape[0] - 1):

        action = model.predict(obs, deterministic=True)[0]

        obs, reward, terminated, truncated, info = test_env.step(action)
        # if terminated or truncated:
        #     test_env.reset()
        render = test_env.render()
        history.append(copy.deepcopy(render)) 

    return pd.DataFrame(history, index=test_data[0].index)

    # "PPO", seed, train_data, test_data, parameters, contender_name, contenders, logger
def train(model_type, seed, train_data, test_data, trade_data, parameters, contender_name, contenders, logger, turbulence):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # Note, kwargs from ensemble ipynb
    train_env = Monitor(TradingEnv(train_data, parameters, parameters['starting_cash'], turbulence))
    if model_type == "A2C":
        # model = A2C("MlpPolicy", train_env, verbose=0, seed=seed, n_steps= 5, ent_coef= 0.005, learning_rate= 0.0007) #ent_coef=parameters["ent_coef"])
        model = A2C("MlpPolicy", train_env, verbose=0, seed=seed) #ent_coef=parameters["ent_coef"])
    else:
        # model = PPO("MlpPolicy", train_env, verbose=0, seed=seed, ent_coef= 0.01, n_steps= 2048, learning_rate= 0.00025, batch_size= 128) #ent_coef=parameters["ent_coef"])
        model = PPO("MlpPolicy", train_env, verbose=0, seed=seed) #ent_coef=parameters["ent_coef"])

    return train_model(model_type, model, train_data, test_data, trade_data, parameters["training_rounds_per_contender"], contender_name, contenders, logger, parameters, turbulence)

def train_model(model_type, model, train_data, test_data, trade_data, training_rounds_per_contender, contender_name, contenders, logger, parameters, turbulence):

    model = model
    best_score = -100
    score = 0

    logger.print_out("Started training a model")
    
    for i in range(training_rounds_per_contender):

        if model_type == "A2C":
            model.learn(total_timesteps=parameters["timesteps_between_check_A2C"], progress_bar=False, reset_num_timesteps=False)
        else:
            model.learn(total_timesteps=parameters["timesteps_between_check_PPO"], progress_bar=False, reset_num_timesteps=False)
        test_history = test_model(model, test_data, parameters, parameters['starting_cash'], turbulence)
        sharpe, _ = get_sharpe_and_volatility(test_history, 'portfolio_value')
        if parameters['validation_parameter'] == 'sharpe':
            score = sharpe
        elif parameters["validation_parameter"] == "last":
            score = i + 1
        else:
            score = test_history.iloc[-1]["portfolio_value"] 
        if score > best_score:
            model.save(contender_name)
            best_score = score

        if parameters["verbose"] == True:
            logger.print_out(f"    - Ended scoring round {i + 1}/{training_rounds_per_contender} with score {score:.2f}, ending value: {test_history.iloc[-1]['portfolio_value'] :.2f}, trading value: {test_model(model, trade_data, parameters, parameters['starting_cash'], turbulence, True).iloc[-1]['portfolio_value']:.2f}, trading sharpe: {get_sharpe_and_volatility(test_model(model, trade_data, parameters, parameters['starting_cash'], turbulence, True), 'portfolio_value')[0]:.2f}")

        # logger.print_out(f"    - Ended training round {i + 1}/{training_rounds_per_contender} with score {best_score:.2f}")

    contenders.append({
        "model": contender_name,
        "score": round(float(best_score), 2)
    })

    logger.print_out(f"- Ended training with score {best_score:.2f}")