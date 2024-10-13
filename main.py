from stable_baselines3 import PPO
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
import json

def main():

    '''
    Loops 3 months long:
    - Train from 9-6 months ago -> 20 PPOs, ordered by validation (6-3 months ago) score
        - Train each one just until overfitting by testing each one between rounds of training
    - Trade 3-0 months ago with best from training
    '''

    multiprocessing.set_start_method('spawn')
    manager = multiprocessing.Manager()
    multiprocessing_cores = multiprocessing.cpu_count() - 1
    if len(sys.argv) > 1:
        multiprocessing_cores = int(sys.argv[1])

    # Parameters ---------------------

    starting_month = dt.date(year=2016, month=1, day=1)
    ending_month = dt.date(year=2020, month=6, day=1)
    # starting_month = dt.date(year=2024, month=1, day=1)
    # ending_month = dt.date(year=2024, month=8, day=1)

    train_months = 12
    validation_months = 3
    trade_months = 54
    num_PPO_contenders = 24
    num_A2C_contenders = 0
    training_rounds_per_contender = 5
    starting_cash = 1000000
    ent_coef = 0.01

    # ---------------------------------

    cash = starting_cash

    # Instatiate run folder and parameters file
    run_start_time = dt.datetime.now()
    run_folder_name = "runs/" + run_start_time.strftime('%Y-%m-%d-%H-%M-%S')
    ModelTools.make_dir(run_folder_name)
    with open(f"{run_folder_name}/parameters.json", 'w') as f:
        json.dump({
            "starting_month": starting_month.strftime('%Y-%m-%d'),
            "ending_month": ending_month.strftime('%Y-%m-%d'),
            "train_months": train_months,
            "validation_months": validation_months,
            "trade_months": trade_months,
            "num_PPO_contenders": num_PPO_contenders,
            "num_A2C_contenders": num_A2C_contenders,
            "training_rounds_per_contender": training_rounds_per_contender,
            "starting_cash": starting_cash,
            "ent_coef": ent_coef
        }, f)

    # Main Loop
    total_months = math.ceil((ending_month - starting_month).days / 30.44)
    for trade_window_start in [starting_month + pd.DateOffset(months=i) for i in range(0, total_months, trade_months)]:

        # Figure out monthly windows for trading, testing, and training
        train_window_start = trade_window_start - pd.DateOffset(months=validation_months + train_months)
        train_window_end = trade_window_start - pd.DateOffset(months=validation_months, days=1)
        validation_window_start = train_window_start + pd.DateOffset(months=train_months)
        validation_window_end = trade_window_start - pd.DateOffset(days=1)
        trade_window_end = trade_window_start + pd.DateOffset(months=trade_months) - pd.DateOffset(days=1)
        
        # Printout dates
        print(f"\nStarting round with trading window [{trade_window_start.strftime('%Y-%m-%d')}, {trade_window_end.strftime('%Y-%m-%d')}],")
        if validation_months == 0:
            print("Validation window is equal to training window,")
        else:
            print(f"Validation window [{validation_window_start.strftime('%Y-%m-%d')}, {validation_window_end.strftime('%Y-%m-%d')}],")
        print(f"Training window [{train_window_start.strftime('%Y-%m-%d')}, {train_window_end.strftime('%Y-%m-%d')}]\n")

        # Get train, test, and trade data
        train_data = StockData.get_consecutive_months(starting_month=train_window_start, num_months=train_months)
        if validation_months == 0:
            test_data = train_data
        else:
            test_data = StockData.get_consecutive_months(starting_month=validation_window_start, num_months=validation_months)
        trade_data = StockData.get_consecutive_months(starting_month=trade_window_start, num_months=trade_months)

        # Instantiate trade window folder
        trade_window_folder_name = f"{run_folder_name}/{trade_window_start.strftime('%Y-%m-%d')}"
        make_dir(trade_window_folder_name + "/models")

        print(f"Finished initializing, beginning to train contenders.\n")
        training_start_time = dt.datetime.now()

        # Train our PPO contenders using multiprocessing 
        processes = []
        contenders = manager.list()
        for i in range(int(num_A2C_contenders) + int(num_PPO_contenders)):
            if i < int(num_A2C_contenders):
                contender_name = f"{trade_window_folder_name}/models/A2C_{i}"
                p = multiprocessing.Process(target=ModelTools.train_A2C, args=(int(random.random() * 100000), train_data, test_data, training_rounds_per_contender, contender_name, contenders, ent_coef))
            else:
                contender_name = f"{trade_window_folder_name}/models/PPO_{i}"
                p = multiprocessing.Process(target=ModelTools.train_PPO, args=(int(random.random() * 100000), train_data, test_data, training_rounds_per_contender, contender_name, contenders, ent_coef))
            p.start()
            processes.append(p)

            if len(processes) >= multiprocessing_cores:
                for p in processes:
                    p.join()
                processes = []
        for p in processes:
            p.join()
        contenders = list(contenders)
        contenders.sort(key=lambda x: x['score'], reverse=True)

        print(f"\nFinished training contenders in {(dt.datetime.now() - training_start_time).seconds} seconds.\n")

        # Print PPO contenders for debugging
        for p in contenders:
            print(p)

        # Get best PPO contender and trade with them
        print(f"\nStarting trading with model with score {contenders[0]['score']:.2f}")
        trade_window_history = ModelTools.test_model(PPO.load(contenders[0]['model']), trade_data, cash)
        ModelTools.write_history_to_file(trade_window_history, f"{trade_window_folder_name}/trade_window_history")

        # Update running balance
        cash = trade_window_history.iloc[-1]['portfolio_value'] # Assume bot sells off all stocks at end of 3 month period
        print(f"- Finished trading, new running cash total: {cash:.2f}\n\n\n")

    combined_history = ModelTools.combine_trade_window_histories(run_folder_name)

    ModelTools.print_stats_from_history(combined_history)

if __name__ == "__main__":
    main()