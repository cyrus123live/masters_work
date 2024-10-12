import ModelTools

run_directory = 'runs/2024-10-11-22-08-51'

history = ModelTools.combine_trade_window_histories(run_directory)

ModelTools.plot_history(history)
ModelTools.get_stats_from_history(history)