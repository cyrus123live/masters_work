import ModelTools
import sys
import yfinance as yf

# run_directory = 'runs/2024-10-11-22-08-51'
# run_directory = 'runs/2024-10-12-00-10-18' # Ensemble window run, 12 contenders with 5 training rounds each
# run_directory = 'runs/2024-10-12-19-45-01' # 2.75 half a2c half PPO
# run_directory = 'runs/2024-10-13-01-53-43' # Baseline one PPO one round 
# run_directory = 'runs/2024-10-13-11-23-15' # 24 PPO's take on 2024
# run_directory = 'runs/2024-10-13-12-13-13' # One Trade Window, 24 PPOs
# run_directory = 'runs/2024-10-13-04-05-06' # Failed run on server (16 cpus took ~18 hours cost 10$)
run_directory = 'runs/2024-10-13-14-28-03' # 24 PPO 24 A2C obliterate ensemble window

# run_directory = 'runs/2024-10-14-16-07-35' # Test with coninuous k for buy and sell
# run_directory = 'runs/2024-10-14-16-29-07' # Test with discrete
# run_directory = 'runs/2024-10-14-17-32-01'
# run_directory = 'runs/imported_runs/2024-10-13-17-33-24' # Test on windows, 16/16 1/1
# run_directory = 'runs/imported_runs/2024-10-14-00-30-29' # Test on DO, 10/10 3/3

# run_directory = 'runs/2024-10-15-18-59-24' # Daily test #1
# run_directory = 'runs/2024-10-16-09-03-40' # Daily test #2, discrete and sharpe

# run_directory = 'runs/2024-10-14-20-40-23' # continous test
run_directory = 'runs/2024-10-16-14-03-32' # removing morning and noon restriction

if len(sys.argv) > 1:
    run_directory = sys.argv[1]

history = ModelTools.combine_trade_window_histories(run_directory)

ModelTools.print_parameters(run_directory)
ModelTools.print_stats_from_history(history)
ModelTools.plot_history(history)
