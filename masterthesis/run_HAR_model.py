from config import instance_path
from HAR_Model import *

# Data import
df_m = pd.read_csv(instance_path.path_input + '/' + 'RealizedMeasures03_10.csv',
                   index_col=0)
df_m.DATE = df_m.DATE.values
df_m.DATE = pd.to_datetime(df_m.DATE, format='%Y%m%d')

# Create HAR Model
har_20_T = HARModel(
    df=df_m,
    future=20,
    lags=[4, 20],
    feature='RV_s',
    semi_variance=True,
    period_train=list([pd.to_datetime('20030910', format='%Y%m%d'), pd.to_datetime('20081001', format='%Y%m%d')]),
    period_test=list([pd.to_datetime('20090910', format='%Y%m%d'), pd.to_datetime('20100301', format='%Y%m%d')]))

har_20_T.run_complete_model()  # generates some warning messages (does not really matter)
