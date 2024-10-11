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

def main():

    '''
    Loops 3 months long:
    - Train from 9-6 months ago -> 20 PPOs, ordered by validation (6-3 months ago) score
        - Train each one just until overfitting by testing each one between rounds of training
    - Trade 3-0 months ago with best from training
    '''

    run_start_time = dt.datetime.now()
    run_folder_name = "runs/" + run_start_time.strftime('%Y-%m-%d-%H-%M-%S')

    # multiprocessing.set_start_method('spawn', force=True)
    multiprocessing_cores = 16

    # Backtest
    starting_month = dt.date(year=2016, month=1, day=1)
    ending_month = dt.date(year=2020, month=7, day=1)

    train_months = 3
    validation_months = 3
    trade_months = 3
    num_contenders = 16
    training_rounds_per_contender = 10
    starting_cash = 10000000

    cash = starting_cash
    history = pd.DataFrame()

    total_months = math.ceil((ending_month - starting_month).days / 30.44)

    for trade_window_start in [starting_month + pd.DateOffset(months=i) for i in range(0, total_months, trade_months)]:

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

        manager = multiprocessing.Manager()
        results = manager.dict()
        # for i in range(num_contenders):
        #     print(f"\nTraining Contender: {i+1}/{num_contenders}\n")
        #     PPO_Contenders.append(ModelTools.train_PPO(train_data, test_data, training_rounds_per_contender))
        #     print(f"\nContender Trained, Validation Score: {PPO_Contenders[-1]['score']:.2f}\n")

        print(f"Finished initializing, beginning to train contenders.\n\n")

        processes = []
        for i in range(int(num_contenders)):
            contender_name = f"{trade_window_folder_name}/models/{i}"
            p = multiprocessing.Process(target=ModelTools.train_PPO, args=(train_data, test_data, training_rounds_per_contender, contender_name, results))
            p.start()
            processes.append(p)

            # Limit the number of concurrent processes
            if len(processes) >= multiprocessing_cores:
                for p in processes:
                    p.join()
                processes = []

        # Join any remaining processes
        for p in processes:
            p.join()
        
        print(f"Finished training contenders in {(dt.datetime.now() - time).seconds} seconds.")
        # PPO_Contenders = list(PPO_Contenders)

        results = dict(results)
        print(results)
        
        PPO_Contenders = [{
            "model": PPO.load(f"{trade_window_folder_name}/models/{i}.zip"),
            "score": results[f"{trade_window_folder_name}/models/{i}"]["score"]
        }for i in range(num_contenders)]

        # Get best PPO contender and trade with them
        PPO_Contenders.sort(key=lambda x: x['score'], reverse=True)
        print(f"\nStarting Trading with model with score {PPO_Contenders[0]['score']:.2f}\n")
        history = pd.concat([history, ModelTools.test_model(PPO_Contenders[0]['model'], trade_data, cash)])

        # Update running balance
        cash = history.iloc[-1]['portfolio_value'] # Assume bot sells off all stocks at end of 3 month period
        print(f"\nFinished Trading, new running cash total: {cash:.2f}\n\n\n")

    ModelTools.get_stats_from_history(history)
    ModelTools.write_history_to_file(history, f"{run_folder_name}/result")
    ModelTools.plot_history(history)

if __name__ == "__main__":
    main()