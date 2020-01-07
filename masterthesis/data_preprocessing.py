from config import instance_path
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import numpy as np

# data loading :: this should not happen here though (only for verification
# purposes

df_m = pd.read_csv(instance_path.path_input + '/' + 'RealizedMeasures03_10.csv',
                   index_col=0)
df_m.DATE = df_m.DATE.values
df_m.DATE = pd.to_datetime(df_m.DATE, format='%Y%m%d')


class HARModel_test:

    subset = ['DATE', 'RV_s', 'RSV_s_plus', 'RSV_s_minus']

    def __init__(self,
                 df):
        self.df = df
        self.subset = None
        self.lag = None
        self.output = None

    def lagged_average(self, target, period):
        lag_vector = np.array([0])
        data_lag = self.df[[target]]
        for i in range(period, data_lag.shape[0]):
            new = np.mean(data_lag.iloc[(i-period):i])
            lag_vector = np.append(lag_vector, new)
        lag_vector = np.delete(lag_vector, 0)
        help_vector = np.full(period, np.nan)
        lag_vector = np.append(help_vector, lag_vector)
        lag_vector = pd.DataFrame(lag_vector)
        lag_vector.set_index([lag_vector.index.values - 1])
        self.lag = lag_vector

    # def future_average(self):
    #     output = pd.DataFrame(np.nan,
    #                           index=range())
    #
    #
    # def har_data_frame_generator(self):


class HARModel:

    """
    cross validation
    """
    def __init__(self,
                 df,
                 lags=[1, 5, 20],  # evt liste daraus machen
                 features=['RV_s', 'RSV_s_plus', 'RSV_s_minus'],
                 period_train=['2003-09-01', '2008-10-01'],
                 period_test=['2008-10-02', '2010-12-31'],
                 semi_variance=False):  #if semi_variance is True the model
                                        #with semi variance is estimated
                                        #what about signed jumps? neglect?
        self.df = df
        self.lags = lags
        self.features = features
        self.period_train = period_train
        self.period_test = period_test
        self.semi_variance = semi_variance
        self.prediction_train = None
        self.prediction_test = None
        self.estimators = None
        self.accuracy = None


instance = HARModel(df_m)
instance.semi_variance