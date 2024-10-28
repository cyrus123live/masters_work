import StockData
import TradingEnv
import ModelTools
import json
from stable_baselines3 import A2C
import datetime as dt

run_folder = "runs/2024-10-28-13-10-52"
model_name = f"{run_folder}/2016-01-01/models/A2C_0"

with open(f"{run_folder}/parameters.json", 'r') as f:
    parameters = json.load(f)

model = A2C.load(model_name)

ModelTools.plot_history(ModelTools.test_model(model, StockData.get_consecutive_months(dt.datetime(year=17, month=1, day=1), 36, parameters), parameters), parameters)