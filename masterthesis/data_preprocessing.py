from config import instance_path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn')

# data loading :: this should not happen here though (only for verification
# purposes

df_m = pd.read_csv(instance_path.path_input + '/' + 'RealizedMeasures03_10.csv',
                   index_col=0)
df_m.DATE = df_m.DATE.values
df_m.DATE = pd.to_datetime(df_m.DATE, format='%Y%m%d')


class HARModel:

    def __init__(self,
                 df,
                 future=1,
                 lags=[4, 20],
                 feature='RV_s',  # depending on the data frame it can be adapted
                 period_train=['20030910', '20081001'],  #pd.to_datetime('20090110', format='%Y%m%d')  # format to date
                 period_test=['20081002', '20101231'],
                 semi_variance=False):
        self.df = df
        self.future = future
        self.lags = lags
        self.feature = feature
        self.period_train = period_train
        self.period_test = period_test
        self.semi_variance = semi_variance
        self.prediction_train = None  # vector (or excel export)
        self.prediction_test = None  # vector (or excel export
        self.estimators = None  # table
        self.accuracy = None  # dictionary
        self.output_df = None  # DataFrame

    def lag_average(self):

        df = self.df[['DATE', self.feature]]  # shift DATE !!!
        df['RV_t'] = df[self.feature].shift(-1)
        df['RV_t_4'] = df[self.feature].rolling(window=4).mean()

        rw20 = self.df[self.feature].rolling(window=20).mean()

        df['RV_t_20'] = list(
                ((self.lags[1] * rw20) - (self.lags[0] * df.RV_t_4)) /
                (self.lags[1] - self.lags[0])
        )

        df['DATE'] = df.DATE.shift(-1)

        df = df.dropna().reset_index(drop=True)

        df.drop(['RV_s'], axis=1, inplace=True)

        assert round(df.RV_t[0:4].mean() - df.RV_t_4[4], 4) == 0, 'Error'

        self.output_df = df

    def future_average(self):
        self.lag_average()
        df = self.output_df.copy()
        df_help = pd.DataFrame()

        for i in range(self.future):
            df_help[str(i)] = df.RV_t.shift(-(1 + i))

        df['future'] = df_help.mean(axis=1)

        self.output_df = df

        assert (self.output_df.RV_t[1:(self.future+1)].mean()
                - self.output_df.future[0] == 0), 'Error'

    # def generate_complete_data_set(self):
    #     if self.semi_variance:
    #         self.future_average()
    #         df = self.output_df.copy()
    #         df


instance_test = HARModel(
    df=df_m,
    future=1,
    lags=[4, 20],
    feature='RV_s')

instance_test.lag_average()
instance_test.future_average()
instance_test.output_df.head()

