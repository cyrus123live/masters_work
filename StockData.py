import yfinance as yf
from datetime import datetime
from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
import random
import requests
import datetime as dt
from dotenv import load_dotenv
import os
from stockstats import StockDataFrame as Sdf
import stockstats
import copy

def calculate_turbulence(data, parameters):
    df_price_pivot = pd.DataFrame(index = data[0].index)
    for i, tic in enumerate(data):
        df_price_pivot[parameters["tickers"][i]] = tic["close"]

    start = 252  # Start after one trading year
    turbulence_index = [0] * start
    unique_date = df_price_pivot.index.unique()
    count = 0

    # Loop over each date starting from 'start'
    for i in range(start, len(unique_date)):
        current_price = df_price_pivot.loc[unique_date[i]].values
        hist_price = df_price_pivot.loc[unique_date[0:i]]

        # Compute the historical covariance matrix
        cov_temp = hist_price.cov()
        # Handle potential singular matrix
        cov_inv = np.linalg.pinv(cov_temp)
        # Compute the difference between current price and historical mean
        current_temp = current_price - hist_price.mean().values
        # Calculate Mahalanobis distance
        temp = current_temp.dot(cov_inv).dot(current_temp.T)
        if temp > 0:
            count += 1
            if count > 2:
                turbulence_temp = temp
            else:
                # Avoid large outlier due to initial calculations
                turbulence_temp = 0
        else:
            turbulence_temp = 0
        turbulence_index.append(turbulence_temp)

    # Create a DataFrame with the turbulence index
    turbulence_index = pd.DataFrame({
        'datadate': df_price_pivot.index,
        'turbulence': turbulence_index
    })
    return turbulence_index

def invert_ticker(df):
    inverse = copy.deepcopy(df)
    inverse["close"] = df["close"].pct_change() * -1 + 1
    inverse["close"] = inverse["close"].cumprod() * df["close"].iloc[0]
    inverse.iloc[0] = df.iloc[0]
    return inverse


def calculate_td_combo(df):
    indicator = pd.DataFrame(index=df.index)

    setup_cat = [0 for _ in range(len(df))]
    countdown_completed_cat = [0 for _ in range(len(df))]
    setup_count = [0 for _ in range(len(df))]
    countdown_count = [0 for _ in range(len(df))]

    for i, close in enumerate(df["close"]):
        if i < 4: continue
        if close < df["close"].iloc[i - 4]:
            setup_count[i] = setup_count[i - 1] + 1
        else:
            setup_cat[i] = 0
            setup_count[i] = 0
            countdown_count[i] = 0
            countdown_completed_cat[i] = 0

        if setup_count[i] > 0:

            if close <= df["low"].iloc[i - 2]: 
                countdown_count[i] = countdown_count[i - 1] + 1
            else:
                countdown_count[i] = countdown_count[i - 1]

            if countdown_count[i] >= 13: countdown_completed_cat[i] = 1

        if setup_count[i] >= 9:
            setup_cat[i] = 1

    indicator["setup_cat"] = setup_cat
    indicator["countdown_completed_cat"] = countdown_completed_cat
    indicator["setup_count"] = setup_count
    indicator["countdown_count"] = countdown_count

    return indicator


def process_data(data):

    # print(data)

    data.index.name = "date"
    stock = Sdf.retype(data.copy())

    processed_data = pd.DataFrame(index=data.index)
    processed_data['close'] = data['close']
    processed_data['open'] = data['open']
    processed_data['low'] = data['low']
    processed_data['high'] = data['high']
    processed_data['volume'] = data['volume']

    # td_combo = calculate_td_combo(processed_data)

    # processed_data["setup_cat"] = td_combo["setup_cat"]
    # processed_data["countdown_completed_cat"] = td_combo["countdown_completed_cat"]
    # processed_data["setup_count"] = td_combo["setup_count"]
    # processed_data["countdown_count"] = td_combo["countdown_count"]

    processed_data['macd'] = stock['macd']
    processed_data['rsi'] = stock['rsi_30']
    processed_data['cci'] = stock['cci_30']
    processed_data['adx'] = stock['dx_30']

    # processed_data['log-return'] = stock['log-ret']
    # processed_data['rsi'] = stock['rsi']
    # processed_data['stoch_rsi'] = stock['stochrsi']
    # processed_data['atr'] = stock['atr']
    # processed_data['mfi'] = stock['mfi']
    # processed_data['supertrend_ub'] = stock['supertrend_ub']
    # processed_data['supertrend_lb'] = stock['supertrend_lb']
    # processed_data['chop'] = stock['chop']

    # stockstats.set_dft_window('macd', (6, 13, 5))
    # processed_data["macd"] = stock['macd']
    # processed_data["macds"] = stock['macds']
    # processed_data["macdh"] = stock['macdh']

    # processed_data["Close"] = data["close"]
    # processed_data["Change"] = data["close"].diff()
    # processed_data["D_HL"] = data["high"] - data["low"]

    # For Spread Calculation --------------
    # processed_data["Close_Mean"] = processed_data["Close"].rolling(window=20).mean()
    # processed_data["Close_STD"] = processed_data["Close"].rolling(window=20).std()
    # -------------------------------------

    for feature in processed_data.columns:
        # Calculate rolling mean and std
        rolling_mean = processed_data[feature].rolling(window=20).mean()
        rolling_std = processed_data[feature].rolling(window=20).std()

        # Normalize the feature
        processed_data[f'{feature}_normalized'] = (processed_data[feature] - rolling_mean) / rolling_std

        # Min-Max Scaling to range -1 to 1 using rolling window 
        # rolling_min = processed_data[f'{feature}_Normalized'].rolling(window=20).min()
        # rolling_max = processed_data[f'{feature}_Normalized'].rolling(window=20).max()
        # processed_data[f'{feature}_Scaled'] = -1 + 2 * (processed_data[f'{feature}_Normalized'] - rolling_min) / (rolling_max - rolling_min)

    # processed_data.ffill(inplace=True)
    processed_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    processed_data.dropna(inplace=True)

    # return processed_data.iloc[40 - processed_data.index[0].minute:]
    return processed_data

def get_min_max_values():
    historical_data = get_consecutive_months(dt.datetime(year=2000, month=1, day=1), 120) # get 2000-2010 data 
    for c in historical_data.columns:
        print(c)
        print("Max: ", historical_data[c].max()) 
        print("Min: ", historical_data[c].min())
    

def get_daily_csv(year, month, ticker):
    df = pd.read_csv(f"stock_data/{ticker}_daily.csv", parse_dates=['timestamp'], index_col='timestamp')
    return df[df.index.strftime("%Y-%m") == f"20{year:02d}-{month:02d}"]
    

def get_month_csv(year, month, ticker):
    folder_name = f"{ticker}_data"

    data = pd.read_csv(f"stock_data/{folder_name}/20{year:02d}-{month:02d}.csv", index_col="timestamp")
    data.index = pd.to_datetime(data.index)

    return data


def get_month_hourly(year, month):
    return process_data(
        get_month_csv(year, month).resample('h').agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        ).dropna()
    )

def get_month(starting_month, i, parameters):
    output = []
    year = (starting_month + pd.DateOffset(months=i)).year % 100
    month = (starting_month + pd.DateOffset(months=i)).month
    for ticker in parameters["tickers"]:
        if parameters["t"] == "daily":
            frames = []
            frames.append(get_daily_csv(year, month, ticker).dropna().iloc[::-1])
            i_year = year
            i_month = month
            for i in range(5):
                i_month -= 1
                if i_month < 1:
                    i_month = 12
                    i_year -= 1
                frames.append(get_daily_csv(i_year, i_month, ticker).dropna().iloc[::-1])
            t_data = process_data(pd.concat(frames).iloc[::-1])
            output.append(t_data[t_data.index.month == month])
        elif parameters["t"] == "half-hourly":
            frames = []
            if month == 1:
                frames.append(get_month_csv(year - 1, 12, ticker).resample('30min').agg(
                    open=('open', 'first'),
                    high=('high', 'max'),
                    low=('low', 'min'),
                    close=('close', 'last'),
                    volume=('volume', 'sum')
                ).dropna())
            else:
                frames.append(get_month_csv(year, month - 1, ticker).resample('30min').agg(
                    open=('open', 'first'),
                    high=('high', 'max'),
                    low=('low', 'min'),
                    close=('close', 'last'),
                    volume=('volume', 'sum')
                ).dropna())
            frames.append(get_month_csv(year, month, ticker).resample('30min').agg(
                open=('open', 'first'),
                high=('high', 'max'),
                low=('low', 'min'),
                close=('close', 'last'),
                volume=('volume', 'sum')
            ).dropna())

            processed = process_data(pd.concat(frames))
            output.append(processed[processed.index.month == month])
        else: 
            output.append(process_data(get_month_csv(year, month, ticker).dropna().iloc[::-1]).dropna())

    if parameters["shorting"]:
        inverses = []
        [inverses.append(invert_ticker(df)) for df in output]
        output.extend(inverses)

    return output

def get_random_month_not_2008():
    year = random.randint(0, 23)
    while year == 8:
        year = random.randint(0, 23)
    return get_month(year, random.randint(1, 12))

def get_random_month_2020s():
    return get_month(random.randint(20, 23), random.randint(1, 12))

def get_random_month_2020s_hourly():
    return get_month_hourly(random.randint(20, 23), random.randint(1, 12))

def get_random_month():
    return get_month(random.randint(0, 23), random.randint(1, 12))

def get_test_data():
    frames = []
    for i in range(1, 9):
        frames.append(get_month(24, i))
    return pd.concat(frames)

def get_random_train_data(num_months):
    frames = []
    for i in range(num_months):
        frames.append(get_random_month())
    return pd.concat(frames)


def get_consecutive_months(starting_month, num_months, parameters):
    frames = [[] for i in range(len(parameters["tickers"]))]
    if parameters["shorting"]:
        frames = [[] for i in range(len(parameters["tickers"]) * 2)]
    for i in range(num_months):
        data = get_month(starting_month, i, parameters)
        for i, d in enumerate(data):
            frames[i].append(d)   

    # Remove all data where index doesn't exist for every df
    data = [pd.concat(f) for f in frames]
    common_index = data[0].index
    for df in data[1:]:
        common_index = common_index.intersection(df.index)

    data = [df.loc[common_index] for df in data]

    return data

def get_year(year):
    frames = []
    for i in range(1, 13):
        frames.append(get_month(year, i))
    return pd.concat(frames)

def get_year_hourly(year):
    frames = []
    for i in range(1, 13):
        frames.append(get_month_hourly(year, i))
    return pd.concat(frames)

def get_pre_2020():
    frames = []
    for i in range(0, 20):
        frames.append(get_year(i))
    return pd.concat(frames)

def get_pre_2020_hourly():
    frames = []
    for i in range(0, 20):
        frames.append(get_year_hourly(i))
    return pd.concat(frames)

def get_day(year, month, day):
    month = get_month(year, month)
    return month[month.index.day == day]

def get_current_data():

    # Note: provides a week of data
    prices = yf.Ticker("SPY").history(period='max', interval='1m', prepost=True)
    prices["close"] = prices["Close"]
    prices["open"] = prices["Open"]
    prices["low"] = prices["Low"]
    prices["high"] = prices["High"]
    prices["volume"] = prices["Volume"]

    prices = prices[prices.index.day == int(dt.datetime.today().strftime("%d"))] # Get current date data

    return process_data(prices)
    
def get_current_alpaca():

    load_dotenv()
    api_key = os.getenv("API_KEY")
    api_secret_key = os.getenv("API_SECRET_KEY")

    url = "https://data.alpaca.markets/v2/stocks/bars?symbols=spy&timeframe=1Min&limit=3000&adjustment=raw&feed=sip&sort=asc"
    headers = {"accept": "application/json", "APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret_key}

    response = requests.get(url, headers=headers) 

    data = pd.DataFrame.from_dict(response.json()['bars']['SPY'])

    data['time'] = pd.to_datetime(data['t'], format='%Y-%m-%dT%H:%M:%SZ')
    data['time'] = data['time'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    data = data.set_index('time')

    data['close'] = data['c']
    data['high'] = data['h']
    data['low'] = data['l']

    return process_data(data)