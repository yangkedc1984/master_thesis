from Result_Tables_Graphs import *

for i in [1, 5, 20]:
    result_instance = ResultOutput(forecast_period=i)
    result_instance.run_all(save_=False, save_plots=True)
