from stable_baselines3 import A2C, PPO, DDPG
from TradingEnv import TradingEnv
import StockData
import ModelTools

import math
import pandas as pd
import datetime as dt
import multiprocessing
import os
from ModelTools import make_dir
import sys
import random
from sb3_contrib import RecurrentPPO
import json

def main():

    if sys.argv[2] is not None:
        indicators = sys.argv[2].split(",")
    else:
        # indicators = ["close_normalized", 'macd_normalized', 'rsi_normalized', 'cci_normalized', "adx_normalized"]
        indicators = ["close", "low", "high", "volume", "setup_cat", "countdown_completed_cat", "setup_count", "countdown_count", "log-return", "rsi", "stoch_rsi", "atr", "mfi", "supertrend_ub", "supertrend_lb", "chop", "macd", "macds", "macdh"]

    print(indicators)

    parameters = {
        "starting_month": "2020-3", # Half-hourly data doesn't work for 2020-01 and below (this parameter is for trading month)
        "ending_month": "2024-9", # note: can do 2022-2024 for good and bad conditions
        "train_months": 1,
        "test_months": 1,
        "trade_months": 1,
        "num_recurrent_ppo": 0,
        "num_ddpg": 0, # Note: DDPG doesn't work for discrete yet 
        "num_ppo": 0,
        "num_a2c": 32, 
        "test_before_train": False,
        "training_rounds_per_contender": 1,
        "timesteps_per_round_Recurrent_PPO": "data_len",
        "timesteps_per_round_DDPG": 10000, 
        "timesteps_per_round_PPO": 50000, 
        "timesteps_per_round_A2C": 25000, 
        "starting_cash": 1000000,
        "buy_sell_action_space": "discrete", 
        "ent_coef": 0, # note: disabled for a2c
        "shorting": False,
        'validation_parameter': "sharpe",
        "indicators": indicators,
        # "indicators": ["close", 'macd', 'rsi', 'cci', "adx"],
        # "indicators": ["close", "setup_cat", "countdown_completed_cat", "setup_count", "countdown_count"],
        # "indicators": ["close", "low", "high", "volume", "setup_cat", "countdown_completed_cat", "setup_count", "countdown_count", "log-return", "rsi", "stoch_rsi", "atr", "mfi", "supertrend_ub", "supertrend_lb", "chop", "macd", "macds", "macdh"],
        # "indicators": ["close_normalized"],
        # "indicators": ["close"],
        "fees": 0.0005, # Works for discrete only (note: ensemble uses 0.001, MEXC charges 0.01% = 0.0001, 2017 assumes 0.25% = 0.0025)
        "use_turbulence": False,
        "turbulence_threshold": 200, # Doesn't work for crypto yet
        "t": "minutely",
        # "tickers": ["BTCUSDT", "ETHUSDT", "XRPUSDT", "BNBUSDT", "TRXUSDT"],
        "tickers": ["spy"],
        # "tickers": ["BTCUSDT"],
        # "tickers": ['AXP', 'AAPL', 'VZ', 'BA', 'CAT', 'JPM', 'CVX', 'KO', 'DIS', 'DD', 'XOM', 'HD', 'INTC', 'IBM', 'JNJ', 'MCD', 'MRK', 'MMM', 'NKE', 'PFE', 'PG', 'UNH', 'RTX', 'WMT', 'WBA', 'MSFT', 'CSCO', 'TRV', 'GS', 'V'],
        "verbose": True,
        "cores": 4
    }

    multiprocessing.set_start_method('spawn')
    manager = multiprocessing.Manager()

    runs_start_time = dt.datetime.now()
    for run in range(int(sys.argv[1])):

        cash = parameters["starting_cash"]

        # Instantiate run folder and parameters file
        run_start_time = dt.datetime.now()
        run_folder_name = "runs/" + runs_start_time.strftime('%Y-%m-%d-%H-%M-%S') + "/" + str(run)
        ModelTools.make_dir(run_folder_name)
        logger = ModelTools.Logger(run_folder_name)
        with open(f"{run_folder_name}/parameters.json", 'w') as f:
            json.dump(parameters, f)

        # Load turbulence file
        if parameters["use_turbulence"]:
            turbulence = pd.read_csv("turbulence.csv")
            turbulence["turbulence"] = pd.to_numeric(turbulence["turbulence"])
            turbulence["datadate"] = pd.to_datetime(turbulence["datadate"])

        # Main Loop
        starting_month = dt.datetime(year=int(parameters["starting_month"].split("-")[0]), month=int(parameters["starting_month"].split("-")[1]), day=1)
        ending_month = dt.datetime(year=int(parameters["ending_month"].split("-")[0]), month=int(parameters["ending_month"].split("-")[1]), day=1)
        total_months = math.ceil((ending_month - starting_month).days / 30.44)
        for trade_window_start in [starting_month + pd.DateOffset(months=i) for i in range(0, total_months, parameters["trade_months"])]:

            trade_window_start_time = dt.datetime.now()

            # Figure out monthly windows for trading, testing, and training
            if parameters["test_before_train"]:
                validation_window_start = trade_window_start - pd.DateOffset(months=parameters["train_months"] + parameters["test_months"])
                validation_window_end = trade_window_start - pd.DateOffset(months=parameters["train_months"], days=1)
                train_window_start = trade_window_start - pd.DateOffset(months=parameters["train_months"])
                train_window_end = trade_window_start - pd.DateOffset(days=1)
                trade_window_end = trade_window_start + pd.DateOffset(months=parameters["trade_months"]) - pd.DateOffset(days=1)

            else:
                train_window_start = trade_window_start - pd.DateOffset(months=parameters["test_months"] + parameters["train_months"])
                train_window_end = trade_window_start - pd.DateOffset(months=parameters["test_months"], days=1)
                validation_window_start = train_window_start + pd.DateOffset(months=parameters["train_months"])
                validation_window_end = trade_window_start - pd.DateOffset(days=1)
                trade_window_end = trade_window_start + pd.DateOffset(months=parameters["trade_months"]) - pd.DateOffset(days=1)
            
            # Printout dates
            logger.print_out(f"\nStarting round with trading window [{trade_window_start.strftime('%Y-%m-%d')}, {trade_window_end.strftime('%Y-%m-%d')}],")
            if parameters["test_months"] == 0:
                logger.print_out("Validation window is equal to training window,")
            else:
                logger.print_out(f"Validation window [{validation_window_start.strftime('%Y-%m-%d')}, {validation_window_end.strftime('%Y-%m-%d')}],")
            logger.print_out(f"Training window [{train_window_start.strftime('%Y-%m-%d')}, {train_window_end.strftime('%Y-%m-%d')}]\n")

            # Get train, test, and trade data
            train_data = StockData.get_consecutive_months(starting_month=train_window_start, num_months = parameters["train_months"], parameters = parameters)
            if parameters["test_months"] == 0:
                test_data = train_data
            else:
                test_data = StockData.get_consecutive_months(starting_month=validation_window_start, num_months = parameters["test_months"], parameters = parameters)
            try:
                trade_data = StockData.get_consecutive_months(starting_month=trade_window_start, num_months = parameters["trade_months"], parameters = parameters)
            except:
                trade_data = test_data

            if not parameters["use_turbulence"]:
                turbulence = pd.DataFrame(index = trade_data[0].index)
                turbulence["datadate"] = turbulence.index
                turbulence["turbulence"] = [-1 for _ in range(len(turbulence))]

            # Instantiate trade window folder
            trade_window_folder_name = f"{run_folder_name}/{trade_window_start.strftime('%Y-%m-%d')}"
            make_dir(trade_window_folder_name + "/models")

            logger.print_out(f"Finished initializing, beginning to train contenders.\n")
            training_start_time = dt.datetime.now()

            if parameters["cores"] == 1:
                contenders = []
                for i in range(int(parameters["num_a2c"]) + int(parameters["num_ppo"]) + int(parameters["num_recurrent_ppo"]) + int(parameters["num_ddpg"])):
                    seed = int(random.random() * 100000)
                    if i < int(parameters["num_a2c"]):
                        contender_name = f"{trade_window_folder_name}/models/A2C_{i}"
                        ModelTools.train("A2C", seed, train_data, test_data, trade_data, parameters, trade_window_folder_name, contender_name, contenders, logger, turbulence)
                    elif i < int(parameters["num_a2c"]) + int(parameters["num_recurrent_ppo"]):
                        contender_name = f"{trade_window_folder_name}/models/Recurrent_PPO_{i}"
                        ModelTools.train("Recurrent_PPO", seed, train_data, test_data, trade_data, parameters, trade_window_folder_name, contender_name, contenders, logger, turbulence)
                    elif i < int(parameters["num_a2c"]) + int(parameters["num_recurrent_ppo"]) + int(parameters["num_ddpg"]):
                        contender_name = f"{trade_window_folder_name}/models/DDPG_{i}"
                        ModelTools.train("DDPG", seed, train_data, test_data, trade_data, parameters, trade_window_folder_name, contender_name, contenders, logger, turbulence)
                    else:
                        contender_name = f"{trade_window_folder_name}/models/PPO_{i}"
                        ModelTools.train("PPO", seed, train_data, test_data, trade_data, parameters, trade_window_folder_name, contender_name, contenders, logger, turbulence)

            # Train our contenders using multiprocessing 
            else:
                processes = []
                contenders = manager.list()
                for i in range(int(parameters["num_a2c"]) + int(parameters["num_ppo"]) + int(parameters["num_recurrent_ppo"]) + int(parameters["num_ddpg"])):
                    seed = int(random.random() * 100000)
                    if i < int(parameters["num_a2c"]):
                        contender_name = f"{trade_window_folder_name}/models/A2C_{i}"
                        p = multiprocessing.Process(target=ModelTools.train, args=("A2C", seed, train_data, test_data, trade_data, parameters, trade_window_folder_name, contender_name, contenders, logger, turbulence))
                    elif i < int(parameters["num_a2c"]) + int(parameters["num_recurrent_ppo"]):
                        contender_name = f"{trade_window_folder_name}/models/Recurrent_PPO_{i}"
                        p = multiprocessing.Process(target=ModelTools.train, args=("Recurrent_PPO", seed, train_data, test_data, trade_data, parameters, trade_window_folder_name, contender_name, contenders, logger, turbulence))
                    elif i < int(parameters["num_a2c"]) + int(parameters["num_recurrent_ppo"]) + int(parameters["num_ddpg"]):
                        contender_name = f"{trade_window_folder_name}/models/DDPG_{i}"
                        p = multiprocessing.Process(target=ModelTools.train, args=("DDPG", seed, train_data, test_data, trade_data, parameters, trade_window_folder_name, contender_name, contenders, logger, turbulence))
                    else:
                        contender_name = f"{trade_window_folder_name}/models/PPO_{i}"
                        p = multiprocessing.Process(target=ModelTools.train, args=("PPO", seed, train_data, test_data, trade_data, parameters, trade_window_folder_name, contender_name, contenders, logger, turbulence))
                    p.start()
                    processes.append(p)

                    if len(processes) >= parameters["cores"]:
                        for p in processes:
                            p.join()
                        processes = []
                for p in processes:
                    p.join()
                contenders = list(contenders)
                
            contenders.sort(key=lambda x: x['score'], reverse=True)
            logger.print_out(f"\nFinished training contenders in {(dt.datetime.now() - training_start_time).seconds} seconds.\n")

            # Print contenders for debugging
            for p in contenders:
                logger.print_out(p)

            # Get best contender and trade with them
            logger.print_out(f"\nStarting trading with model with score {contenders[0]['score']:.2f}")
            model_type = "PPO"
            if "Recurrent_PPO" in contenders[0]['model']:
                model = PPO.load(contenders[0]['model'])
                model_type = "Recurrent_PPO"
            elif "A2C" in contenders[0]['model']:
                model = A2C.load(contenders[0]['model'])
            elif "PPO" in contenders[0]['model']:
                model = PPO.load(contenders[0]['model'])
            elif "DDPG" in contenders[0]['model']:
                model = DDPG.load(contenders[0]['model'])

            trade_window_history = ModelTools.test_model(model, trade_data, parameters, cash, turbulence, True, model_type)
            ModelTools.write_history_to_file(trade_window_history, f"{trade_window_folder_name}/trade_window_history")

            # Update running balance
            cash = trade_window_history.iloc[-1]['portfolio_value'] # Assume bot sells off all stocks at end of 3 month period
            logger.print_out(f"- Finished trading, new running cash total: {cash:.2f}")

            logger.print_out(f"\nFinished date window, total time: {(dt.datetime.now() - trade_window_start_time).seconds} seconds.\n\n")

        combined_history = ModelTools.combine_trade_window_histories(run_folder_name)

        logger.print_out(f"\nFinished run in {(dt.datetime.now() - run_start_time).seconds} seconds.\n")

        # ModelTools.print_stats_from_history(combined_history, parameters)
        # ModelTools.plot_history(combined_history, parameters)

if __name__ == "__main__":
    main()