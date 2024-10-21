import requests
import random
import pandas as pd
import os
import json
import sys

api_key = os.getenv('ALPHA_KEY')
api_key = "OZTJNCRFTEZJ7O0P"
print(api_key)
ticker = sys.argv[1]

url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={api_key}"
data = json.loads(requests.get(url).content.decode('utf-8'))
df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index").iloc[::-1]

to_save = pd.DataFrame(index = df.index)
df.index.name = "timestamp"
to_save["close"] = df["4. close"].to_list()
to_save["open"] = df["1. open"].to_list()
to_save["high"] = df["2. high"].to_list()
to_save["low"] = df["3. low"].to_list()
to_save["volume"] = df["5. volume"].to_list()

print(to_save)
to_save.to_csv(f"stock_data/{ticker.lower()}_daily.csv")
quit()

ticker = "SH"
ticker_folder_name = ticker.lower() + "_data"
os.makedirs(f"stock_data/{ticker_folder_name}/", exist_ok=True)
for year in range(0, 24):
    for month in range(1, 13):
        time = f"20{year:02d}-{month:02d}"
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={ticker}&interval=1min&adjusted=false&extended_hours=true&apikey={api_key}&datatype=csv&outputsize=full&month={time}'
        data = requests.get(url)
        decoded_content = data.content.decode('utf-8')
        # Open the file for writing
        if decoded_content:
            with open(f'stock_data/{ticker_folder_name}/{time}.csv', 'w') as f:
                f.write(decoded_content)
                
        
for year in range(24, 25):
    for month in range(1, 9):
        time = f"20{year:02d}-{month:02d}"
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={ticker}&interval=1min&adjusted=false&extended_hours=true&apikey={api_key}&datatype=csv&outputsize=full&month={time}'
        data = requests.get(url)
        decoded_content = data.content.decode('utf-8')
        # Open the file for writing
        if decoded_content:
            with open(f'stock_data/{ticker_folder_name}/{time}.csv', 'w') as f:
                f.write(decoded_content)