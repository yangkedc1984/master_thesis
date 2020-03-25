from run_HAR_model import *
from LSTM import *


class AutoRegressionModel:
    def __init__(
        self,
        df: pd.DataFrame,
        future: int = 1,
        ar_lag: int = 1,
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
    ):
        self.data = df
        self.future = future
        self.ar_lag = ar_lag
        self.period_train_ar = period_train
        self.period_test_ar = period_test

        self.data_instance = None
        self.training_set = None
        self.testing_set = None
        self.ar_model = None
        self.prediction_train = None
        self.prediction_test = None
        self.train_accuracy = None
        self.test_accuracy = None

    def prepare_data(self):
        self.data_instance = TimeSeriesDataPreparationLSTM(
            df=self.data,
            future=self.future,
            lag=self.ar_lag,
            standard_scaler=False,
            min_max_scaler=False,
            log_transform=False,
            semi_variance=False,
            jump_detect=True,
            period_train=self.period_train_ar,
            period_test=self.period_test_ar,
        )
        self.data_instance.prepare_complete_data_set()
        self.training_set = self.data_instance.training_set
        self.testing_set = self.data_instance.testing_set

    def estimate_model(self):
        if self.testing_set is None:
            self.prepare_data()

        assert (self.ar_lag == 1) or (
            self.ar_lag == 3
        ), "AR lag-operator should be either 1 or 3"

        if self.ar_lag == 1:
            self.ar_model = smf.ols(formula="future ~ RV", data=self.training_set).fit()

        if self.ar_lag == 3:
            self.ar_model = smf.ols(
                formula="future ~ RV + lag_1 + lag_2", data=self.training_set
            ).fit()

    def predict(self):
        if self.ar_model is None:
            self.estimate_model()

        if self.ar_lag == 1:
            self.prediction_train = self.ar_model.predict(self.training_set[["RV"]])
            self.prediction_test = self.ar_model.predict(self.testing_set[["RV"]])

        if self.ar_lag == 3:
            self.prediction_train = self.ar_model.predict(
                self.training_set[["RV", "lag_1", "lag_2"]]
            )
            self.prediction_test = self.ar_model.predict(
                self.testing_set[["RV", "lag_1", "lag_2"]]
            )

    def make_accuracy(self):
        if self.prediction_train is None:
            self.predict()

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
