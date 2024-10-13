import ModelTools
import sys

# run_directory = 'runs/2024-10-11-22-08-51'
run_directory = 'runs/2024-10-12-00-10-18' # Ensemble window run, 12 contenders with 5 training rounds each

if len(sys.argv) > 1:
    run_directory = sys.argv[1]

history = ModelTools.combine_trade_window_histories(run_directory)

ModelTools.print_parameters(run_directory)
ModelTools.print_stats_from_history(history)
ModelTools.plot_history(history)
