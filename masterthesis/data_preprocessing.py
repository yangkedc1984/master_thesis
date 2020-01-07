from config import path_data_input
import pandas as pd

df_m = pd.read_csv(path_data_input, index_col=0)
