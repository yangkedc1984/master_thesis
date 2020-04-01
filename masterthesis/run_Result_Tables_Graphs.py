from Result_Tables_Graphs import *

for i in [1, 5, 20]:
    result_instance = ResultOutput(forecast_period=i, log_transformation=True)
    result_instance.run_all(save_=True, save_plots=False)

result_instance = ResultOutput(forecast_period=5, log_transformation=True)
result_instance.run_all(save_=True, save_plots=False)
