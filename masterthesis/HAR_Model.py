import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn import metrics
import matplotlib.pyplot as plt

plt.style.use("seaborn")
# add method to implement weighted least squares (instead of OLS) --> results will be closer to the true value


class HARModel:
    def __init__(
        self,
        df,
        future=1,
        lags=[4, 20],
        feature="RV",
        semi_variance=False,
        jump_detect=True,
        log_transformation=False,
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
        self.lags = lags
        self.feature = feature
        self.semi_variance = semi_variance
        self.jump_detect = jump_detect
        self.log_transformation = log_transformation
        self.period_train = period_train
        self.period_test = period_test
        self.training_set = None  # data frames
        self.testing_set = None  # data frames
        self.prediction_train = None  # vector (or excel export)
        self.prediction_test = None  # vector (or excel export)
        self.model = None  # stats model instance
        self.estimation_results = None  # table
        self.test_accuracy = None  # dictionary
        self.train_accuracy = None
        self.output_df = None  # DataFrame (data frame which contains all the features needed for the regression)

    def jump_detection(self):
        df_tmp = self.df.copy()
        df_tmp["threshold"] = df_tmp["RV"].rolling(window=200).std() * 4
        df_tmp.threshold = np.where(df_tmp.threshold.isna(), 1, df_tmp.threshold)
        df_tmp["larger"] = np.where(df_tmp.RV > df_tmp.threshold, True, False)
        df_tmp = df_tmp[df_tmp.larger == False]

        df_tmp.drop(columns={"threshold", "larger"}, axis=1, inplace=True)

        # add unit test
        self.df = df_tmp.copy()

    def log_transform(self):
        df_tmp = self.df.copy()
        df_tmp["RV"] = np.log(df_tmp["RV"])

        self.df = df_tmp.copy()

    def lag_average(self):

        df = self.df[["DATE", self.feature]]
        df["RV_t"] = df[self.feature].shift(-1)
        df["RV_w"] = df[self.feature].rolling(window=self.lags[0]).mean()

        rw20 = self.df[self.feature].rolling(window=self.lags[1]).mean()

        df["RV_m"] = list(
            ((self.lags[1] * rw20) - (self.lags[0] * df.RV_w))
            / (self.lags[1] - self.lags[0])
        )

        df["DATE"] = df.DATE.shift(-1)

        df = df.dropna().reset_index(drop=True)

        df.drop([self.feature], axis=1, inplace=True)

        assert (
            round(df.RV_t[0 : self.lags[0]].mean() - df.RV_w[self.lags[0]], 12) == 0
        ), "Error"

        self.output_df = df

    def future_average(self):

        self.lag_average()
        df = self.output_df.copy()

        df_help = pd.DataFrame()

        for i in range(self.future):
            df_help[str(i)] = df.RV_t.shift(-(1 + i))
        df_help = df_help.dropna()

        df["future"] = df_help.mean(axis=1)

        df = df.dropna().reset_index(drop=True)

        self.output_df = df

        assert (
            self.output_df.RV_t[1 : (self.future + 1)].mean() - self.output_df.future[0]
            == 0
        ), "Error"

    def generate_complete_data_set(self):

        if self.jump_detect:
            self.jump_detection()

        if self.log_transformation:
            self.log_transform()

        if self.semi_variance:
            self.future_average()
            df = self.output_df.copy()

            if self.log_transformation:
                self.df["RSV_plus"] = np.log(self.df["RSV_plus"])
                self.df["RSV_minus"] = np.log(self.df["RSV_minus"])

            df = df.merge(self.df[["DATE", "RSV_plus", "RSV_minus"]], on="DATE")

        else:
            self.future_average()
            df = self.output_df

        self.output_df = df

        assert (
            self.output_df.RV_t.iloc[1 : (self.future + 1)].mean()
            - self.output_df.future[0]
            == 0
        ), "Error"

    def make_testing_training_set(self):
        self.generate_complete_data_set()
        df = self.output_df.copy()

        df_train = df.loc[
            (df.DATE >= self.period_train[0]) & (df.DATE <= self.period_train[1])
        ].reset_index(drop=True)
        df_test = df.loc[
            (df.DATE >= self.period_test[0]) & (df.DATE <= self.period_test[1])
        ].reset_index(drop=True)

        self.training_set = df_train
        self.testing_set = df_test

    def estimate_model(self):
        self.make_testing_training_set()

        if self.semi_variance:
            result = smf.ols(
                formula="future ~ RSV_plus + RSV_minus + RV_w + RV_m",
                data=self.training_set,
            ).fit()
        else:
            result = smf.ols(
                formula="future ~ RV_t + RV_w + RV_m", data=self.training_set
            ).fit()

        self.model = result
        results_robust = result.get_robustcov_results(
            cov_type="HAC", maxlags=2 * (self.future - 1)
        )
        self.estimation_results = results_robust.summary().as_latex()

    def predict_values(self):
        self.estimate_model()
        if self.log_transformation:
            if self.semi_variance:
                self.prediction_train = np.exp(
                    self.model.predict(
                        self.training_set[["RSV_plus", "RSV_minus", "RV_w", "RV_m"]]
                    )
                )
                self.prediction_test = np.exp(
                    self.model.predict(
                        self.testing_set[["RSV_plus", "RSV_minus", "RV_w", "RV_m"]]
                    )
                )
            else:
                self.prediction_train = np.exp(
                    self.model.predict(self.training_set[["RV_t", "RV_w", "RV_m"]])
                )
                self.prediction_test = np.exp(
                    self.model.predict(self.testing_set[["RV_t", "RV_w", "RV_m"]])
                )
        else:
            if self.semi_variance:
                self.prediction_train = self.model.predict(
                    self.training_set[["RSV_plus", "RSV_minus", "RV_w", "RV_m"]]
                )
                self.prediction_test = self.model.predict(
                    self.testing_set[["RSV_plus", "RSV_minus", "RV_w", "RV_m"]]
                )
            else:
                self.prediction_train = self.model.predict(
                    self.training_set[["RV_t", "RV_w", "RV_m"]]
                )
                self.prediction_test = self.model.predict(
                    self.testing_set[["RV_t", "RV_w", "RV_m"]]
                )

    def make_accuracy_measures(self):
        """
        Function that reports the accuracy measures for the out-of-sample and the in-sample prediction.
        Accuracy measures are: RMSE, MAE, MAPE and the R-Squared, Beta and Alpha of the
        Mincer-Zarnowitz Regression (R-Squared should be as high as possible, Beta equal to one and alpha equal to zero)

        :return:
        """
        self.predict_values()

        test_accuracy = {
            "MSE": metrics.mean_squared_error(
                self.testing_set["future"], self.prediction_test
            ),
            "MAE": metrics.mean_absolute_error(
                self.testing_set["future"], self.prediction_test
            ),
            "RSquared": metrics.r2_score(
                self.testing_set["future"], self.prediction_test
            ),
        }
        train_accuracy = {
            "MSE": metrics.mean_squared_error(
                self.training_set["future"], self.prediction_train
            ),
            "MAE": metrics.mean_absolute_error(
                self.training_set["future"], self.prediction_train
            ),
            "RSquared": metrics.r2_score(
                self.training_set["future"], self.prediction_train
            ),
        }

        self.test_accuracy = test_accuracy
        self.train_accuracy = train_accuracy

    def run_complete_model(self):
        self.make_accuracy_measures()

    def make_graph(self):
        self.predict_values()

        plt.figure()
        plt.plot(
            self.training_set.DATE,
            self.training_set.future,
            label="Realized Volatility",
            alpha=0.5,
            color="black",
        )
        plt.plot(
            self.training_set.DATE,
            self.prediction_train,
            label="Predicted Volatility",
            alpha=0.75,
            color="darkgreen",
        )
        plt.legend()
        plt.show()


from run_HAR_model import load_data

df_input = load_data()


class HARModelLogTransformed:
    def __init__(
        self,
        df,
        future=1,
        lags=[4, 20,],
        feature="RV",
        semi_variance=False,
        jump_detect=True,
        log_transformation=False,
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
        self.lags = lags
        self.feature = feature
        self.semi_variance = semi_variance
        self.jump_detect = jump_detect
        self.log_transformation = log_transformation
        self.period_train = period_train
        self.period_test = period_test
        self.training_set = None  # data frames
        self.testing_set = None  # data frames
        self.prediction_train = (
            None  # vector (or excel export) :: we need the output for the Dashboad !!
        )
        self.prediction_test = None  # vector (or excel export)
        self.model = None  # stats model instance
        self.estimation_results = None  # table
        self.test_accuracy = None  # dictionary
        self.train_accuracy = None
        self.output_df = None  # DataFrame (data frame which contains all the features needed for the regression)

    def jump_detection(self):
        df_tmp = self.df.copy()
        df_tmp["threshold"] = df_tmp["RV"].rolling(window=200).std() * 4
        df_tmp.threshold = np.where(df_tmp.threshold.isna(), 1, df_tmp.threshold)
        df_tmp["larger"] = np.where(df_tmp.RV > df_tmp.threshold, True, False)
        df_tmp = df_tmp[df_tmp.larger == False]

        df_tmp.drop(columns={"threshold", "larger"}, axis=1, inplace=True)

        # add unit test
        self.df = df_tmp.copy()

    def lag_average(self, log_transform: bool = False, generate_output: bool = False):

        df_tmp = self.df[["DATE", self.feature]]

        if log_transform:
            df_tmp[self.feature] = np.log(df_tmp[self.feature])

        df_tmp["RV_t"] = df_tmp[self.feature].shift(-1)
        df_tmp["RV_w"] = df_tmp[self.feature].rolling(window=self.lags[0]).mean()

        rolling_sum_month = df_tmp[self.feature].rolling(window=self.lags[1]).sum()
        rolling_sum_week = df_tmp[self.feature].rolling(window=self.lags[0]).sum()

        df_tmp["RV_m"] = (rolling_sum_month - rolling_sum_week) / (
            self.lags[1] - self.lags[0]
        )

        df_tmp["DATE"] = df_tmp.DATE.shift(-1)

        df_tmp = df_tmp.dropna().reset_index(drop=True)

        df_tmp.drop([self.feature], axis=1, inplace=True)

        df_tmp = df_tmp

        # unit test: lag_average()
        df_unit_test = self.df[["DATE", self.feature]]

        if log_transform:
            df_unit_test[self.feature] = np.log(df_unit_test[self.feature])

        df_unit_test = df_unit_test.loc[df_unit_test["DATE"] <= df_tmp["DATE"][0]]

        assert (
            round(
                df_tmp.RV_w[0]
                - np.mean(
                    df_unit_test.RV[(self.lags[1] - self.lags[0]) : self.lags[1]]
                ),
                12,
            )
            + round(
                df_tmp.RV_m[0]
                - np.mean(df_unit_test.RV[0 : (self.lags[1] - self.lags[0])]),
                12,
            )
            == 0
        ), "Error: Lagged average realized volatility computation error"

        if generate_output:
            return df_tmp

    def future_average(self):

        df = self.lag_average(log_transform=False, generate_output=True)

        df_help = pd.DataFrame()

        for i in range(self.future):
            df_help[str(i)] = df.RV_t.shift(-(1 + i))
        df_help = df_help.dropna()

        self.output_df = self.lag_average(
            log_transform=self.log_transformation, generate_output=True
        )

        if self.log_transformation:
            self.output_df["future"] = np.log(df_help.mean(axis=1))
            self.output_df = self.output_df.dropna().reset_index(drop=True)
        else:
            self.output_df["future"] = df_help.mean(axis=1)
            self.output_df = self.output_df.dropna().reset_index(drop=True)

        # unit test: future_average()
        df_unit_test = self.lag_average(log_transform=False, generate_output=True)
        if self.log_transformation:
            test_unit = np.log(np.mean(df_unit_test.RV_t[1 : (self.future + 1)]))
        else:
            test_unit = np.mean(df_unit_test.RV_t[1 : (self.future + 1)])

        assert (
            self.output_df.future[0] - test_unit == 0
        ), "Error: Future average realized volatility computation error"

    def generate_complete_data_set(self):

        if self.jump_detect:
            self.jump_detection()

        if self.semi_variance:
            self.future_average()
            df = self.output_df.copy()

            if self.log_transformation:
                self.df["RSV_plus"] = np.log(self.df["RSV_plus"])
                self.df["RSV_minus"] = np.log(self.df["RSV_minus"])

            df = df.merge(self.df[["DATE", "RSV_plus", "RSV_minus"]], on="DATE")

        else:
            self.future_average()
            df = self.output_df

        self.output_df = df

    def make_testing_training_set(self):
        self.generate_complete_data_set()
        df = self.output_df.copy()

        df_train = df.loc[
            (df.DATE >= self.period_train[0]) & (df.DATE <= self.period_train[1])
        ].reset_index(drop=True)
        df_test = df.loc[
            (df.DATE >= self.period_test[0]) & (df.DATE <= self.period_test[1])
        ].reset_index(drop=True)

        self.training_set = df_train
        self.testing_set = df_test

    def estimate_model(self):
        self.make_testing_training_set()

        if self.semi_variance:
            result = smf.ols(
                formula="future ~ RSV_plus + RSV_minus + RV_w + RV_m",
                data=self.training_set,
            ).fit()
        else:
            result = smf.ols(
                formula="future ~ RV_t + RV_w + RV_m", data=self.training_set
            ).fit()

        self.model = result
        results_robust = result.get_robustcov_results(
            cov_type="HAC", maxlags=2 * (self.future - 1)
        )
        self.estimation_results = results_robust.summary().as_latex()

    def predict_values(self):
        self.estimate_model()
        if self.log_transformation:
            if self.semi_variance:
                self.prediction_train = np.exp(
                    self.model.predict(
                        self.training_set[["RSV_plus", "RSV_minus", "RV_w", "RV_m"]]
                    )
                )
                self.prediction_test = np.exp(
                    self.model.predict(
                        self.testing_set[["RSV_plus", "RSV_minus", "RV_w", "RV_m"]]
                    )
                )
            else:
                self.prediction_train = np.exp(
                    self.model.predict(self.training_set[["RV_t", "RV_w", "RV_m"]])
                )
                self.prediction_test = np.exp(
                    self.model.predict(self.testing_set[["RV_t", "RV_w", "RV_m"]])
                )
        else:
            if self.semi_variance:
                self.prediction_train = self.model.predict(
                    self.training_set[["RSV_plus", "RSV_minus", "RV_w", "RV_m"]]
                )
                self.prediction_test = self.model.predict(
                    self.testing_set[["RSV_plus", "RSV_minus", "RV_w", "RV_m"]]
                )
            else:
                self.prediction_train = self.model.predict(
                    self.training_set[["RV_t", "RV_w", "RV_m"]]
                )
                self.prediction_test = self.model.predict(
                    self.testing_set[["RV_t", "RV_w", "RV_m"]]
                )

    def make_accuracy_measures(self):
        """
        Function that reports the accuracy measures for the out-of-sample and the in-sample prediction.
        Accuracy measures are: RMSE, MAE, MAPE and the R-Squared, Beta and Alpha of the
        Mincer-Zarnowitz Regression (R-Squared should be as high as possible, Beta equal to one and alpha equal to zero)

        :return:
        """
        self.predict_values()

        if self.log_transformation:
            self.testing_set["future"] = np.exp(self.testing_set["future"])
            self.training_set["future"] = np.exp(self.training_set["future"])

        test_accuracy = {
            "MSE": metrics.mean_squared_error(
                self.testing_set["future"], self.prediction_test
            ),
            "MAE": metrics.mean_absolute_error(
                self.testing_set["future"], self.prediction_test
            ),
            "RSquared": metrics.r2_score(
                self.testing_set["future"], self.prediction_test
            ),
        }
        train_accuracy = {
            "MSE": metrics.mean_squared_error(
                self.training_set["future"], self.prediction_train
            ),
            "MAE": metrics.mean_absolute_error(
                self.training_set["future"], self.prediction_train
            ),
            "RSquared": metrics.r2_score(
                self.training_set["future"], self.prediction_train
            ),
        }

        self.test_accuracy = test_accuracy
        self.train_accuracy = train_accuracy

    def run_complete_model(self):
        self.make_accuracy_measures()


model_har_log = HARModelLogTransformed(
    df=df_input,
    future=20,
    lags=[4, 20],
    feature="RV",
    semi_variance=True,
    jump_detect=True,
    log_transformation=True,
    period_train=list(
        [
            pd.to_datetime("20030910", format="%Y%m%d"),
            pd.to_datetime("20091231", format="%Y%m%d"),
        ]
    ),
    period_test=list(
        [
            pd.to_datetime("20100101", format="%Y%m%d"),
            pd.to_datetime("20101231", format="%Y%m%d"),
        ]
    ),
)

model_har = HARModel(
    df=df_input,
    future=20,
    lags=[4, 20],
    feature="RV",
    semi_variance=True,
    jump_detect=True,
    log_transformation=False,
    period_train=list(
        [
            pd.to_datetime("20030910", format="%Y%m%d"),
            pd.to_datetime("20091231", format="%Y%m%d"),
        ]
    ),
    period_test=list(
        [
            pd.to_datetime("20100101", format="%Y%m%d"),
            pd.to_datetime("20101231", format="%Y%m%d"),
        ]
    ),
)

from LSTM import *

lstm_instance = TimeSeriesDataPreparationLSTM(
    df=df_input,
    future=20,
    lag=20,
    standard_scaler=False,
    min_max_scaler=True,
    log_transform=True,
    semi_variance=False,
    jump_detect=True,
    period_train=list(
        [
            pd.to_datetime("20030910", format="%Y%m%d"),
            pd.to_datetime("20091231", format="%Y%m%d"),
        ]
    ),
    period_test=list(
        [
            pd.to_datetime("20100101", format="%Y%m%d"),
            pd.to_datetime("20101231", format="%Y%m%d"),
        ]
    ),
)
lstm_instance.prepare_complete_data_set()

model_har_log.run_complete_model()
model_har.run_complete_model()

# proof that all fits perfectly :: Can you unit test this somewhere? probably best location in Results.py
plt.close()
plt.plot(model_har.training_set.DATE, model_har.training_set.future, label="HAR")
plt.plot(
    model_har_log.training_set.DATE,
    model_har_log.training_set.future,
    "-.",
    label="HAR log",
)
plt.plot(
    lstm_instance.training_set.DATE,
    lstm_instance.back_transformation(
        np.array(lstm_instance.training_set.future).reshape(-1, 1)
    ),
    label="LSTM DATA",
)
plt.legend()

assert (
    round(
        sum(
            (
                lstm_instance.back_transformation(
                    np.array(lstm_instance.testing_set.future).reshape(-1, 1)
                ).reshape(lstm_instance.testing_set.shape[0],)
                - model_har.testing_set.future
            )
        ),
        20,
    )
    == 0
), "Error: Unequal future realized volatility for LSTM- and HAR Model (unequal dependent variable in the models)"
