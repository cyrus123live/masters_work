import requests
import datetime as dt
import pandas as pd
import time
import os

# df = pd.read_csv("stock_data/btc_data.csv", index_col="timestamp")
# df.index = pd.to_datetime(df.index)
# for year in range(df.index[0].year, df.index[-1].year + 1):
#     year_data = df[df.index.year == year]
#     for month in range(year_data.index[0].month, year_data.index[-1].month + 1):
#         month_data = year_data[year_data.index.month == month]
#         month_data.sort_index(ascending=True)
#         month_data.iloc[::-1].to_csv(f'stock_data/btc_data/{month_data.index[0].year}-{month_data.index[0].month:02d}.csv', header=True)
# quit()

def parse_and_save(data, header = False):
    minute_data = pd.DataFrame(data)
    minute_data["time"] = minute_data[0] / 1000.0
    minute_data.index = [dt.datetime.utcfromtimestamp(minute_data.iloc[i]["time"]) for i in range(len(minute_data))]
    parsed_data = pd.DataFrame(index=minute_data.index)
    parsed_data.index.name = "timestamp"
    parsed_data["open"] = minute_data[1]
    parsed_data["close"] = minute_data[4]
    parsed_data["high"] = minute_data[2]
    parsed_data["low"] = minute_data[3]
    parsed_data["volume"] = minute_data[5]
    parsed_data = parsed_data.head(parsed_data.shape[0] -1)
    # path = f'stock_data/btc_data/{parsed_data.index[0].year}-{parsed_data.index[0].month:02d}.csv'
    # if os.path.isfile(path):
    #     header = False
    # else:
    #     header = True
    parsed_data.to_csv('stock_data/btc_data.csv', mode='a', header=header)
    print(parsed_data)

url = 'https://api.binance.com/api/v3/klines'
url = 'https://api.binance.com/api/v3/ticker/price'

print(requests.get(url))
quit()

os.makedirs("stock_data/", exist_ok=True)

day = 0

parse_and_save(requests.get(url, params={
    'symbol': 'BTCUSDT',
    'startTime': int(dt.datetime.timestamp(dt.datetime(year=2018, month=1, day=1) + dt.timedelta(hours=(24 * day) - 8)) * 1000),
    'endTime': int(dt.datetime.timestamp(dt.datetime(year=2018, month=1, day=1) + dt.timedelta(hours=(24 * day) + 4)) * 1000),
    'interval': '1m',
    'limit': '1000' 
}).json(), True)

parse_and_save(requests.get(url, params={
    'symbol': 'BTCUSDT',
    'startTime': int(dt.datetime.timestamp(dt.datetime(year=2018, month=1, day=1) + dt.timedelta(hours=(24 * day) + 4)) * 1000),
    'endTime': int(dt.datetime.timestamp(dt.datetime(year=2018, month=1, day=1) + dt.timedelta(hours=(24 * day) + 16)) * 1000),
    'interval': '1m',
    'limit': '1000' 
}).json())

for day in range(1, (dt.datetime(year=2024, month=9, day=30) - dt.datetime(year=2020, month=1, day=1)).days):

    parse_and_save(requests.get(url, params={
        'symbol': 'BTCUSDT',
        'startTime': int(dt.datetime.timestamp(dt.datetime(year=2018, month=1, day=1) + dt.timedelta(hours=(24 * day) - 8)) * 1000),
        'endTime': int(dt.datetime.timestamp(dt.datetime(year=2018, month=1, day=1) + dt.timedelta(hours=(24 * day) + 4)) * 1000),
        'interval': '1m',
        'limit': '1000' 
    }).json())
    
    parse_and_save(requests.get(url, params={
        'symbol': 'BTCUSDT',
        'startTime': int(dt.datetime.timestamp(dt.datetime(year=2018, month=1, day=1) + dt.timedelta(hours=(24 * day) + 4)) * 1000),
        'endTime': int(dt.datetime.timestamp(dt.datetime(year=2018, month=1, day=1) + dt.timedelta(hours=(24 * day) + 16)) * 1000),
        'interval': '1m',
        'limit': '1000' 
    }).json())

    time.sleep(1)

# final = pd.concat(to_save)
# for year in range(final.index[0].year, final.index[-1].year + 1):
#     year_data = final[final.index.dt.year == year]
#     for month in range(year_data.index[0].month, year_data.index[-1].month + 1):
#         month_data = year_data[year_data.index.dt.month == month]
#         month_data.to_csv(f'stock_data/btc_data/{month_data.index[0].year}-{month_data.index[0].month:02d}.csv', header=True)