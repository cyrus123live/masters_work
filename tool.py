import ModelTools

# run_directory = 'runs/2024-10-11-22-08-51'
run_directory = 'runs/2024-10-12-00-10-18' # Ensemble window run, 12 contenders with 5 training rounds each

history = ModelTools.combine_trade_window_histories(run_directory)

ModelTools.print_parameters(run_directory)
ModelTools.print_stats_from_history(history)
ModelTools.plot_history(history)
