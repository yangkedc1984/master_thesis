"""
Class makes volatility predictions by applying deep learning

plt.hist(np.log(results["har_1_True"].df.RV), bins=100)
        # check optimal distribution of data with given scaler

"""

import pandas as pd
import random


class LSTM:
    def __init__(
        self,
        df,
        future=1,
        lag=20,
        feature="RV",
        semi_variance=False,
        period_train=list(
            [
                pd.to_datetime("20030910", format="%Y%m%d"),
                pd.to_datetime("20080208", format="%Y%m%d"),
            ]
        ),
        period_test=list(
            [
                pd.to_datetime("20080209", format="%Y%m%d"),
                pd.to_datetime("20101231", format="%Y%m%d"),
            ]
        ),
    ):
        self.df = df
        self.future = future
        self.lag = lag
        self.feature = feature
        self.semi_variance = semi_variance
        self.period_train = period_train
        self.period_test = period_test
        # self.scale_min_max = True
        # self.scale_log = True
        # self.scale_standard = True

        # Predefined generated output
        self.training_set = None  # data frames
        self.testing_set = None  # data frames
        self.prediction_train = None  # vector (or excel export)
        self.prediction_test = None  # vector (or excel export)
        self.model = None  # stats model instance
        self.estimation_results = None  # table
        self.test_accuracy = None  # dictionary
        self.train_accuracy = None
        self.future_values = None
        self.historical_values = None
        self.df_processed_data = None
        self.train_matrix = None
        self.train_y = None

    def future_averages(self):
        df = self.df[["DATE", "RV"]]
        for i in range(self.future):
            df["future_{}".format(i + 1)] = df.RV.shift(-(i + 1))
        df = df.dropna()

        help_df = df.drop(["DATE", "RV"], axis=1)

        df_output = df[["DATE", "RV"]]
        df_output["future"] = help_df.mean(axis=1)

        # unit testing
        s = random.randint(0, df_output.shape[0])
        assert (help_df.iloc[s].mean() - df_output.future.iloc[s]) == 0, "Error"

        self.future_values = df_output

    def historical_lag_transformation(self):
        df = self.df[["DATE", "RV"]]
        for i in range((self.lag - 1)):
            df["lag_{}".format(i + 1)] = df.RV.shift(+(i + 1))

        df = df.drop(["RV"], axis=1)

        # add unit test
        self.historical_values = df

    def generate_complete_data_set(self):
        self.future_averages()
        self.historical_lag_transformation()

        data_set_complete = self.future_values.merge(
            self.historical_values, how="right", on="DATE"
        )
        data_set_complete = data_set_complete.dropna()
        data_set_complete.reset_index(drop=True, inplace=True)

        if self.semi_variance:
            df_tmp = self.df[["DATE", "RSV_minus"]]
            data_set_complete = data_set_complete.merge(df_tmp, on="DATE")

        self.df_processed_data = data_set_complete

    def make_testing_training_set(self):
        self.generate_complete_data_set()
        df = self.df_processed_data.copy()

        df_train = df.loc[
            (df.DATE >= self.period_train[0]) & (df.DATE <= self.period_train[1])
        ].reset_index(drop=True)
        df_test = df.loc[
            (df.DATE >= self.period_test[0]) & (df.DATE <= self.period_test[1])
        ].reset_index(drop=True)

        self.training_set = df_train
        self.testing_set = df_test

    def neural_network_input(self):
        self.make_testing_training_set()
        self.train_matrix = self.training_set.drop(columns={"DATE", "future"}).values
        self.train_y = self.training_set[["future"]].values


# to set hyper-parameters we need a class that defines the NN structure (otherwise the whole data wrangling is redone
# each time)
from run_HAR_model import load_data

df = load_data()

lstm_instance = LSTM(df=df)
lstm_instance.generate_complete_data_set()
lstm_instance.make_testing_training_set()
df_train = lstm_instance.training_set
df_test = lstm_instance.testing_set

lstm_instance.neural_network_input()
lstm_instance.train_y.shape
lstm_instance.train_matrix.shape


# class TrainingNetwork(LSTM):  # this class does not help me much probably (good to know that it works though)
#     def __init__(self, kt):
#         LSTM.__init__(
#             self,
#             df=df,
#             future=1,
#             lag=20,
#             feature="RV",
#             semi_variance=True,
#             period_train=list(
#                 [
#                     pd.to_datetime("20030910", format="%Y%m%d"),
#                     pd.to_datetime("20080208", format="%Y%m%d"),
#                 ]
#             ),
#             period_test=list(
#                 [
#                     pd.to_datetime("20080209", format="%Y%m%d"),
#                     pd.to_datetime("20101231", format="%Y%m%d"),
#                 ]
#             ),
#         )
#         self.kt = kt
#
#     # def make_something(self):
#     #     inc = self.future + self.neuron
#     #
#     #     return inc
#
#
# x = TrainingNetwork(3)
# x.kt
# x.generate_complete_data_set()
