from config import *


import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')


df_m = pd.read_csv(path_data_input + '/' + 'RealizedMeasures03_10.csv',
                   index_col=0)
df_m.DATE = df_m.DATE.values
df_m.DATE = pd.to_datetime(df_m.DATE, format='%Y%m%d')

df_m.columns

plt.hist(df_m['RV_abs'], bins=500)
plt.hist(df_m['RV_s'], bins=500)

plt.plot(df_m.DATE, df_m.RV_abs)

