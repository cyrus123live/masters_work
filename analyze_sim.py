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

run_directory = 'runs/2024-10-28-13-37-00' # First successful (sharpe ratio 1.6 up until later part of 2018) daily model, trading SPY with 10 A2C 10 PPO 3/3/3, 1 round of 100000, ensemble obs with discrete action space (occasionally takes a hit (see 2018 10-12 ) due to randomness in relationship between validation and trade score)  (also, massive drawdown in start of 2020 due to lack of turbulence index)
run_directory = 'runs/2024-10-28-16-11-10' # Exact same setup as above but failed twice, exemplifying issue
run_directory = 'runs/2024-10-28-18-03-24' # 2/6/2 with sharpe, very small consistent returns
run_directory = 'runs/2024-10-28-18-26-33' # 2/6/2 backwards with simple returns, trading and testing aren't correlated, lots of flatness
run_directory = 'runs/2024-10-28-19-15-50' # 2/6/2 backwards with sharpe
run_directory = 'runs/2024-10-28-20-56-34' # No test A2C rounds, successful
run_directory = 'runs/2024-10-28-21-36-31' # Trying again to see if it's luck

run_directory = 'runs/2024-10-29-13-37-16' # Note: weird cash infusion here
run_directory = 'runs/2024-10-29-13-48-59' # really good result with 3/3 35000 A2C but saved crazy action vector

run_directory = 'runs/2024-10-29-18-39-34' # First successful portfolio run, Dow Jones 4 A2C's, a little sketchy but overall seems good
# Got this in final interval of 2018: Ended scoring round 1/1 with score 1161106.10, sharpe: 4.61, trading score: 793843.46, trading sharpe: -3.55
# Not so good, got hit with another one: Ended scoring round 1/1 with score 1252885.24, sharpe: 4.95, trading score: 743485.45, trading sharpe: -1.92
run_directory = 'runs/2024-10-29-20-43-55' # A fail with one A2C model, ensemble's hyperparameters

run_directory = 'runs/2024-10-30-12-58-09' # A full copy ensemble run, 1.45 at start of 2019. Finished with 0.91
# Hyperparameters from https://github.com/AI4Finance-Foundation/FinRL-Meta/blob/master/examples/FinRL_Ensemble_StockTrading_ICAIF_2020.ipynb, except timesteps from original ensemble 
# Note: Should try 10,000/10,000 timesteps from there too 

run_directory = 'runs/2024-10-30-17-55-25' # Cleaned up ensemble with ipynb timesteps in addition to kwargs, added turbulence, going crazyyy with 1.65 as of ending interval of 2018
# Mysterious cash injection with unchanging stock array, but this doesn't seem to affect portfolio_value -> stock array was broken
# Ended with 0.8 because turbulence didn't initiate

# Everything is fixed, let's do multiple runs of 1/1:
# run_directory = 'runs/2024-10-31-12-59-00' # First run is an abject failure
# run_directory = 'runs/2024-10-31-13-41-34' # Second run failure
# run_directory = 'runs/2024-10-31-14-12-49' # Third run, started weak but beat buy and hold by the end
# run_directory = 'runs/2024-10-31-15-01-05' # Fourth run, started a failure but came back to 1.18 with -17.5% by end
# run_directory = 'runs/2024-10-31-15-49-29' # Fifth run, 830000 by end of first interval

# run_directory = 'runs/2024-10-31-20-12-58' # Going back to minutely
# run_directory = 'runs/2024-11-02-00-28-38' # BTC
# run_directory = 'runs/2024-11-02-11-18-28' # BTC 1/1/1, need to do it again with more trained models, chose a bad one in 2021-06
# run_directory = 'runs/2024-11-03-22-20-23' # BTC again, this time 16/16 quit early because macbook ran out of space

run_directory = 'runs/2024-11-07-20-24-02' # Crypto portfolio
run_directory = 'runs/2024-11-09-21-22-17'
run_directory = '/users/cyrusparsons/desktop/2024-11-10-08-13-34' # 8000x return in 4 months with 4 crypto portfolio: data must be broken
# run_directory = 'runs/2024-11-10-14-40-03'
# run_directory = 'runs/2024-11-10-16-18-32'
# run_directory = 'runs/2024-11-10-23-19-47'
# run_directory = 'runs/2024-11-11-12-15-30'

run_directory = 'runs/2024-11-14-13-44-00'
run_directory = 'runs/2024-11-14-16-27-53' # Large Crypto portfolio, rediculous returns 
run_directory = 'runs/2024-11-15-10-45-24' # Zero training sanity check BTC only NOTE: error on 2023-07
run_directory = 'runs/2024-11-15-11-23-01' # Zero training sanity check large portfolio NOTE: error on 2020-10
run_directory = 'runs/2024-11-15-11-36-58' # Same as above, different result, 9% return vs buy and hold's 40%

run_directory = 'runs/2024-11-15-13-00-10' # 5000% percent zero training portfolio vs buy and hold's 1100%
# run_directory = 'runs/2024-11-16-16-33-37' # sanity check: random trade data, continuous, Note: Had 2500x return at one point but ended with -17%
# run_directory = 'runs/2024-11-16-17-14-33' # sanity check: random train data, result is almost identical to buy and hold strategy

# Zero training sanity check discrete
# run_directory = 'runs/2024-11-16-01-14-06'
# run_directory = 'runs/2024-11-16-02-40-50'
# run_directory = 'runs/2024-11-16-04-07-23'

# Zero training sanity check continuous
# run_directory = 'runs/2024-11-16-23-07-16'
# run_directory = 'runs/2024-11-17-00-35-44'
# run_directory = 'runs/2024-11-17-02-04-02'

run_directory = 'runs/2024-11-17-16-48-57' # 10x return in one month
run_directory = 'runs/2024-11-17-17-53-56' # half-hourly test, significantly underperforms B&H with 1500 step rounds
run_directory = 'runs/2024-11-17-18-10-14' # Half-hourly shot up at end
run_directory = 'runs/2024-11-17-19-31-50' # Half-hourly consistent high performance
run_directory = 'runs/2024-11-17-21-03-10' # Half-hourly consistent poor performance

# 16 models instead of above's 4
run_directory = 'runs/2024-11-17-22-25-27'
run_directory = 'runs/2024-11-18-03-10-38'
run_directory = 'runs/2024-11-18-07-56-10'

run_directory = 'runs/2024-11-18-21-11-46' # 32 Models

run_directory = '/Volumes/T7 Touch/masters_work backup/2024-11-20-08-41-48'


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
