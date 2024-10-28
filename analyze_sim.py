import ModelTools
import sys
import json
import yfinance as yf

# run_directory = 'runs/2024-10-11-22-08-51'
# run_directory = 'runs/2024-10-12-00-10-18' # Ensemble window run, 12 contenders with 5 training rounds each
# run_directory = 'runs/2024-10-12-19-45-01' # 2.75 half a2c half PPO
# run_directory = 'runs/2024-10-13-01-53-43' # Baseline one PPO one round 
# run_directory = 'runs/2024-10-13-11-23-15' # 24 PPO's take on 2024
# run_directory = 'runs/2024-10-13-12-13-13' # One Trade Window, 24 PPOs
# run_directory = 'runs/2024-10-13-04-05-06' # Failed run on server (16 cpus took ~18 hours cost 10$)
# run_directory = 'runs/2024-10-13-14-28-03' # 24 PPO 24 A2C obliterate ensemble window

# run_directory = 'runs/2024-10-14-16-07-35' # Test with coninuous k for buy and sell
# run_directory = 'runs/2024-10-14-16-29-07' # Test with discrete
# run_directory = 'runs/2024-10-14-17-32-01'
# run_directory = 'runs/imported_runs/2024-10-13-17-33-24' # Test on windows, 16/16 1/1
# run_directory = 'runs/imported_runs/2024-10-14-00-30-29' # Test on DO, 10/10 3/3

# run_directory = 'runs/2024-10-15-18-59-24' # Daily test #1
# run_directory = 'runs/2024-10-16-09-03-40' # Daily test #2, discrete and sharpe

# run_directory = 'runs/2024-10-14-20-40-23' # continous test
# run_directory = 'runs/2024-10-16-14-03-32' # removing morning and noon restriction
# run_directory = 'runs/2024-10-17-16-44-16' # Removing morning and noon restriction, adding fees
# run_directory = 'runs/2024-10-17-19-51-34' # Latest possible train
# run_directory = 'runs/2024-10-17-22-56-26' # 2023-24
# run_directory = 'runs/2024-10-19-00-32-00' # Close only indicator
# run_directory = 'runs/2024-10-19-13-30-25' # First spread experiment 0.01
# run_directory = 'runs/2024-10-19-14-05-51' # spread 0.001
# run_directory = 'runs/2024-10-19-16-13-00' # fees 0.00007
# run_directory = 'runs/2024-10-19-18-13-37' # fees 0.00007 larger test
# run_directory = 'runs/2024-10-20-01-09-15' # fees 0.00007 larger test bring back indicators (seems to have unnatural price spikes)
# run_directory = 'runs/2024-10-20-14-56-20' # Daily test 2020-2021 with RL indicators
# run_directory = 'runs/2024-10-20-18-35-03' # Daily test IBM 2016-2017
# run_directory = 'runs/2024-10-20-20-40-13' # All the indicators, a year to train, 2016-2017 (positive in short bursts
# run_directory = 'runs/2024-10-20-21-09-41' # Sharpe ratio validation, 2 month period, lots of indicators, does ok

# run_directory = 'runs/2024-10-21-19-22-20' # First portfolio test with 4 stocks, lightly negative sloped return
# run_directory = 'runs/2024-10-21-19-35-14' # Portfolio with 6 stocks, profitable until 2020, big max drawdown
run_directory = 'runs/2024-10-22-13-03-43' # Portfolio with just SPY, 0.98 sharpe, with 7.5 annual return and -9 max drawdown
# run_directory = 'runs/2024-10-3-20-28-32'
# run_directory = "runs/2024-10-24-17-42-09"
# run_directory = "runs/2024-10-24-18-15-21"
# run_directory = 'runs/2024-10-24-21-36-25' # 2016-2018 portfolio is worst performing asset with 3 month discrete, close only
run_directory = 'runs/2024-10-24-22-19-04'
run_directory = 'runs/2024-10-24-22-46-44'
run_directory = 'runs/2024-10-24-23-16-42' # Terrible spy performance
run_directory = 'runs/2024-10-27-14-00-55'
run_directory = 'runs/2024-10-27-21-57-19' # First gam test (fail)
run_directory = 'runs/2024-10-28-11-43-27' # PPO's trained in one long round each, very flat and positive result

if len(sys.argv) > 1:
    run_directory = sys.argv[1]

history = ModelTools.combine_trade_window_histories(run_directory)
with open(f"{run_directory}/parameters.json", 'r') as f:
    parameters = json.loads(f.readline())

# Parse history closes from string
history["closes"] = [[float(i) for i in history["closes"].iloc[i].replace("[", "").replace("]", "").strip().split(",")] for i in range(len(history["closes"]))]

ModelTools.print_parameters(run_directory)
ModelTools.print_stats_from_history(history, parameters)
ModelTools.plot_history(history, parameters)
