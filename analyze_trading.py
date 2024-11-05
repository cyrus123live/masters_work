import requests
import json
from stable_baselines3 import PPO
import matplotlib.pyplot as plt 
import pandas as pd
import datetime as dt
from stable_baselines3 import A2C
import TradingEnv
from stable_baselines3.common.monitor import Monitor
import numpy as np
import StockData
import ModelTools

MODEL_NAME = "PPO_109"
FOLDER_NAME = "2024-11-04"

# Returns a history dataframe
# def test_model_manually(model, test_data = StockData.get_current_data(), starting_cash = 1000000):

#     history = []
#     k = starting_cash / test_data.iloc[0]["Close"]
#     cash = starting_cash
#     held = 0
#     for i in range(test_data.shape[0]):

#         data = test_data.iloc[i]
#         # obs = np.array(test_data[test_data.filter(regex='_Scaled$').columns].iloc[i].tolist() + [np.clip(2 * held / k - 1, -1, 1), np.clip(2 * cash / starting_cash - 1, -1, 1)])
#         obs = np.array(test_data[["Close_Normalized", "Change_Normalized", "D_HL_Normalized"]].iloc[i].tolist() + [held / k, cash / starting_cash])
#         # obs = np.array(test_data[["Close_Normalized", "MACD_Normalized", "RSI_Normalized", "CCI_Normalized", "ADX_Normalized"]].iloc[i].tolist() + [held / k, cash / starting_cash])

#         action = model.predict(obs, deterministic=True)[0][0]

#         if action < 0 and cash == 0:
#             cash += held * data["Close"]
#             held = 0
#         elif action > 0 and cash > 0:
#             to_buy = cash / data["Close"]
#             cash = 0
#             held = to_buy

#         history.append({"portfolio_value": cash + held * data["Close"], "close": data["Close"], "cash": cash, "held": held})

#     return pd.DataFrame(history, index=test_data.index)


def plot_result(result):

    # result.index = [dt.datetime.fromtimestamp(i) for i in result.index]
    # model = A2C.load("/root/RLTrader/models/" + MODEL_NAME)

    result.index = [dt.datetime.fromtimestamp(t).astimezone(tz=dt.timezone(dt.timedelta(hours=-4))) for t in result["Time"]]
    # result["Close"] = result["Close"].shift(-1)
    # print(result.columns)
    print(result)

    # sim = test_model_manually(PPO.load("PPO_109.zip"))

    to_plot = pd.DataFrame(index=result.index)
    to_plot['close'] = result["Close"] / result.iloc[0]["Close"]
    to_plot['portfolio'] = (result["Cash"] + result["Held"] * result["Close"])/ (result.iloc[0]["Cash"] + result.iloc[0]["Held"] * result.iloc[0]["Close"])

    figure = plt.figure()
    p = figure.add_subplot()

    [p.axvline(x = i, color = '#33ff33') for i in result[(result['Bought'] == True) & (result["Missed Buy"] == False)].index]
    [p.axvline(x = i, color = '#ff0000') for i in result[(result['Sold'] == True) & (result["Missed Sell"] == False)].index]
    [p.axvline(x = i, color = '#ddffdd') for i in result[result['Missed Buy'] == True].index]
    [p.axvline(x = i, color = '#ffdddd') for i in result[result['Missed Sell'] == True].index]

    p.plot(to_plot['close'], label="Stock Movement")
    p.plot(to_plot['portfolio'], label="Portfolio Value")
    p.legend()

    plt.show()

def main():
    result = pd.DataFrame.from_dict(requests.get(f"http://104.131.87.187:5000/minutely_json/{FOLDER_NAME}").json())
    plot_result(result)
    # ModelTools.plot_history(test_model_manually(PPO.load("PPO_109")))

if __name__ == "__main__":
    main()