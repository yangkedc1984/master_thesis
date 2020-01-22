"""
Class makes volatility predictions by applying deep learning
"""
from sklearn import metrics
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


class DataPreparationLSTM:
    def __init__(
        self,
        df: pd.DataFrame,
        future: int = 1,
        lag: int = 20,
        feature: str = "RV",
        semi_variance: bool = False,
        jump_detect: bool = True,
        log_transform: bool = True,
        min_max_scaler: bool = True,
        standard_scaler: bool = False,
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
        self.jump_detect = jump_detect
        self.log_transform = log_transform
        self.min_max_scaler = min_max_scaler
        self.standard_scaler = standard_scaler
        self.period_train = period_train
        self.period_test = period_test

        # Predefined generated output
        self.training_set = None  # data frames
        self.testing_set = None  # data frames
        self.future_values = None
        self.historical_values = None
        self.df_processed_data = None
        self.train_matrix = None
        self.train_y = None
        self.test_matrix = None
        self.test_y = None

    def jump_detection(self):
        df_tmp = self.df.copy()
        df_tmp["threshold"] = df_tmp["RV"].rolling(window=200).std() * 4
        df_tmp.threshold = np.where(df_tmp.threshold.isna(), 1, df_tmp.threshold)
        df_tmp["larger"] = np.where(df_tmp.RV > df_tmp.threshold, True, False)
        df_tmp = df_tmp[df_tmp.larger == False]

        df_tmp.drop(columns={"threshold", "larger"}, axis=1, inplace=True)

        # unit test
        self.df = df_tmp.copy()

    def data_scaling(self):

        assert (
            self.min_max_scaler + self.standard_scaler <= 1
        ), "Multiple scaling methods selected"

        if self.log_transform:
            self.df.RV = np.log(self.df.RV)
            if self.semi_variance:
                self.df.RSV_plus = np.log(self.df.RSV_plus)
                self.df.RSV_minus = np.log(self.df.RSV_minus)

        if self.min_max_scaler:
            s = MinMaxScaler()
            self.df.RV = s.fit_transform(self.df.RV.values.reshape(-1, 1))
            if self.semi_variance:
                self.df.RSV_plus = s.fit_transform(
                    self.df.RSV_plus.values.reshape(-1, 1)
                )
                self.df.RSV_minus = s.fit_transform(
                    self.df.RSV_minus.values.reshape(-1, 1)
                )

        if self.standard_scaler:
            self.df.RV = normalize(self.df.RV.values.reshape(-1, 1))
            if self.semi_variance:
                self.df.RSV_plus = normalize(self.df.RSV_plus.values.reshape(-1, 1))
                self.df.RSV_minus = normalize(self.df.RSV_minus.values.reshape(-1, 1))

    def future_averages(self):
        df = self.df[["DATE", "RV"]].copy()
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
        df = self.df[["DATE", "RV"]].copy()
        for i in range((self.lag - 1)):
            df["lag_{}".format(i + 1)] = df.RV.shift(+(i + 1))

        df = df.drop(["RV"], axis=1)

        # add unit test
        self.historical_values = df

    def generate_complete_data_set(self):

        if self.jump_detect:
            self.jump_detection()

        self.data_scaling()

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

    def prepare_complete_data_set(self):
        self.make_testing_training_set()


class TrainLSTM:
    def __init__(
        self,
        training_set,
        testing_set,
        activation=tf.nn.elu,
        epochs=20,
        learning_rate=0.01,
        network_architecture={"Layer1": 20, "Layer2": 3, "Layer3": 5, "Layer4": 12},
    ):
        self.training_set = training_set
        self.testing_set = testing_set
        self.activation = activation
        self.network_architecture = network_architecture
        self.epochs = epochs
        self.learning_rate = learning_rate

        # Predefined output
        self.train_matrix = None
        self.train_y = None
        self.test_matrix = None
        self.test_y = None
        self.fitted_model = None
        self.prediction_train = None
        self.prediction_test = None
        self.test_accuracy = None
        self.train_accuracy = None
        self.fitness = None

    def reshape_input_data(self):
        self.train_matrix = self.training_set.drop(columns={"DATE", "future"}).values
        self.train_y = self.training_set[["future"]].values

        self.test_matrix = self.testing_set.drop(columns={"DATE", "future"}).values
        self.test_y = self.testing_set[["future"]].values

        n_features = 1

        train_shape_rows = self.train_matrix.shape[0]
        train_shape_columns = self.train_matrix.shape[1]

        self.train_matrix = self.train_matrix.reshape(
            (train_shape_rows, train_shape_columns, n_features)
        )

        test_shape_rows = self.test_matrix.shape[0]
        test_shape_columns = self.train_matrix.shape[1]

        self.test_matrix = self.test_matrix.reshape(
            (test_shape_rows, test_shape_columns, n_features)
        )

    def train_lstm(self):
        if self.train_matrix is None:
            self.reshape_input_data()

        tf.reset_default_graph()
        m = tf.keras.models.Sequential()
        m.add(
            tf.keras.layers.LSTM(
                self.network_architecture["Layer1"],
                activation=self.activation,
                return_sequences=True,
                input_shape=(int(self.train_matrix.shape[int(1)]), int(1)),
            )
        )
        if self.network_architecture["Layer2"] > 0:
            if self.network_architecture["Layer3"] > 0:
                if self.network_architecture["Layer4"] > 0:
                    m.add(
                        tf.keras.layers.LSTM(
                            self.network_architecture["Layer2"],
                            activation=self.activation,
                            return_sequences=True,
                        )
                    )
                    m.add(
                        tf.keras.layers.LSTM(
                            self.network_architecture["Layer3"],
                            activation=self.activation,
                            return_sequences=True,
                        )
                    )
                    m.add(
                        tf.keras.layers.LSTM(
                            self.network_architecture["Layer4"],
                            activation=self.activation,
                        )
                    )
                else:
                    m.add(
                        tf.keras.layers.LSTM(
                            self.network_architecture["Layer2"],
                            activation=self.activation,
                            return_sequences=True,
                        )
                    )
                    m.add(
                        tf.keras.layers.LSTM(
                            self.network_architecture["Layer3"],
                            activation=self.activation,
                        )
                    )
            else:
                m.add(
                    tf.keras.layers.LSTM(
                        self.network_architecture["Layer2"], activation=self.activation
                    )
                )
        m.add(tf.keras.layers.Dense(1, activation="linear"))

        o = tf.keras.optimizers.Adam(
            lr=self.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=None,
            decay=0.0,
            amsgrad=False,
        )

        m.compile(optimizer=o, loss=tf.keras.losses.logcosh)

        m.fit(self.train_matrix, self.train_y, epochs=self.epochs, verbose=1)

        self.fitted_model = m

    def predict_lstm(self):
        if self.fitted_model is None:
            self.train_lstm()

        self.prediction_train = self.fitted_model.predict(self.train_matrix)
        self.prediction_test = self.fitted_model.predict(self.test_matrix)

    def make_accuracy_measures(self):
        if self.prediction_test is None:
            self.predict_lstm()

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
        self.fitness = 1 / self.train_accuracy["MSE"]
