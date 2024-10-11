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

def main():

    '''
    Loops 3 months long:
    - Train from 9-6 months ago -> 20 PPOs, ordered by validation (6-3 months ago) score
        - Train each one just until overfitting by testing each one between rounds of training
    - Trade 3-0 months ago with best from training
    '''

    # Parameters ---------------------

    # Backtest
    starting_month = dt.date(year=2017, month=1, day=1)
    ending_month = dt.date(year=2017, month=2, day=1)

    train_months = 1
    validation_months = 1
    trade_months = 1
    num_contenders = 2
    training_rounds_per_contender = 1
    starting_cash = 10000000

    # ---------------------------------

    cash = starting_cash
    history = pd.DataFrame()

    run_start_time = dt.datetime.now()
    run_folder_name = "runs/" + run_start_time.strftime('%Y-%m-%d-%H-%M-%S')

    manager = multiprocessing.Manager()
    multiprocessing_cores = multiprocessing.cpu_count() - 1
    if len(sys.argv) > 1:
        multiprocessing_cores = int(sys.argv[1])

    # Main Loop
    total_months = math.ceil((ending_month - starting_month).days / 30.44)
    for trade_window_start in [starting_month + pd.DateOffset(months=i) for i in range(0, total_months, trade_months)]:

        
        # Instantiate trade window folder
        trade_window_folder_name = f"{run_folder_name}/{trade_window_start.strftime('%Y-%m-%d')}"
        make_dir(trade_window_folder_name + "/models")


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

        time = dt.datetime.now()

        print(f"Finished initializing, beginning to train contenders.\n\n")

        # Train our PPO contenders using multiprocessing 
        processes = []
        PPO_Contenders = manager.list()
        for i in range(int(num_contenders)):
            contender_name = f"{trade_window_folder_name}/models/{i}"
            p = multiprocessing.Process(target=ModelTools.train_PPO, args=(train_data, test_data, training_rounds_per_contender, contender_name, PPO_Contenders))
            p.start()
            processes.append(p)

            if len(processes) >= multiprocessing_cores:
                for p in processes:
                    p.join()
                processes = []
        for p in processes:
            p.join()
        PPO_Contenders = list(PPO_Contenders)
        
        print(f"\nFinished training contenders in {(dt.datetime.now() - time).seconds} seconds.\n")

        # Print PPO contenders for debugging
        for p in PPO_Contenders:
            print(p)

        # Get best PPO contender and trade with them
        PPO_Contenders.sort(key=lambda x: x['score'], reverse=True)
        print(f"\nStarting Trading with model with score {PPO_Contenders[0]['score']:.2f}")
        history = pd.concat([history, ModelTools.test_model(PPO.load(PPO_Contenders[0]['model']), trade_data, cash)])

        # Update running balance
        cash = history.iloc[-1]['portfolio_value'] # Assume bot sells off all stocks at end of 3 month period
        print(f"- Finished Trading, new running cash total: {cash:.2f}\n\n\n")

    ModelTools.get_stats_from_history(history)
    ModelTools.write_history_to_file(history, f"{run_folder_name}/result")
    ModelTools.plot_history(history)

if __name__ == "__main__":
    main()