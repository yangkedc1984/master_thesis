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
import matplotlib.pyplot as plt


# Take out all the links within a Class (this might lead to strange behaviour)


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
        self.train_matrix = None
        self.train_y = None
        self.test_matrix = None
        self.test_y = None
        self.future_values = None
        self.historical_values = None
        self.df_processed_data = None
        self.applied_scaler_features = None
        self.applied_scaler_targets = None

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
            self.applied_scaler_features = s
            self.df.RV = s.fit_transform(self.df.RV.values.reshape(-1, 1))
            if self.semi_variance:
                self.df.RSV_plus = s.fit_transform(
                    self.df.RSV_plus.values.reshape(-1, 1)
                )
                self.df.RSV_minus = s.fit_transform(
                    self.df.RSV_minus.values.reshape(-1, 1)
                )

        if self.standard_scaler:  # implement back transformation method for this
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

        self.future_averages()  # future values have to be computed before the targets are engineered

        self.future_values.future = np.log(self.future_values.future)
        s_targets = MinMaxScaler()
        self.applied_scaler_targets = s_targets
        self.future_values.future = s_targets.fit_transform(
            self.future_values.future.values.reshape(-1, 1)
        )

        self.data_scaling()  # data scaling after future value generation
        self.historical_lag_transformation()

        # merging the two data sets
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

    def back_transformation(self, input_data):
        return np.exp(self.applied_scaler_targets.inverse_transform(input_data))

    def reshape_input_data(self):
        if self.training_set is None:
            self.prepare_complete_data_set()

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


class TrainLSTM:
    def __init__(
        self,
        training_set,
        testing_set,
        activation=tf.nn.elu,
        epochs=20,
        learning_rate=0.001,
        layer_one=20,
        layer_two=15,
        layer_three=8,
        layer_four=4,
    ):
        self.training_set = training_set
        self.testing_set = testing_set
        self.activation = activation
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.layer_one = int(layer_one)
        self.layer_two = int(layer_two)
        self.layer_three = int(layer_three)
        self.layer_four = int(layer_four)

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

        # tf.reset_default_graph()
        m = tf.keras.models.Sequential()
        m.add(
            tf.keras.layers.LSTM(
                self.layer_one,
                activation=self.activation,
                return_sequences=True,
                input_shape=(int(self.train_matrix.shape[int(1)]), int(1)),
            )
        )
        if self.layer_two > 0:
            if self.layer_three > 0:
                if self.layer_four > 0:
                    m.add(
                        tf.keras.layers.LSTM(
                            self.layer_two,
                            activation=self.activation,
                            return_sequences=True,
                        )
                    )
                    m.add(
                        tf.keras.layers.LSTM(
                            self.layer_three,
                            activation=self.activation,
                            return_sequences=True,
                        )
                    )
                    m.add(
                        tf.keras.layers.LSTM(
                            self.layer_four, activation=self.activation,
                        )
                    )
                else:
                    m.add(
                        tf.keras.layers.LSTM(
                            self.layer_two,
                            activation=self.activation,
                            return_sequences=True,
                        )
                    )
                    m.add(
                        tf.keras.layers.LSTM(
                            self.layer_three, activation=self.activation,
                        )
                    )
            else:
                m.add(tf.keras.layers.LSTM(self.layer_two, activation=self.activation))
        m.add(tf.keras.layers.Dense(1, activation="linear"))

        o = tf.keras.optimizers.Adam(
            lr=self.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=None,
            decay=0.0,  # self.learning_rate/ self.epochs
            amsgrad=False,
        )

        m.compile(
            optimizer=o, loss=tf.keras.losses.logcosh
        )  # choose: R-MSE, MAE, logCosh

        m.fit(
            self.train_matrix,
            self.train_y,
            epochs=self.epochs,
            verbose=1,
            # validation_data=(self.test_matrix, self.test_y),  # use it to choose optimal number of epochs
        )

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
        self.fitness = (1 / self.test_accuracy["MSE"]) + (
            1 / self.train_accuracy["MSE"]
        )

    def make_performance_plot(self, show_testing_sample=False):
        if show_testing_sample:
            plt.close()
            fig, axs = plt.subplots(3)
            axs[0].plot(
                self.testing_set.DATE,
                self.testing_set.future,
                label="Realized Volatility",
                alpha=0.5,
                color="black",
                lw=0.5,
            )
            axs[0].plot(
                self.testing_set.DATE,
                self.prediction_test,
                label="Prediction",
                alpha=0.8,
                lw=1,
            )
            axs[0].legend()
            axs[1].plot(
                self.testing_set.future,
                self.prediction_test,
                "o",
                alpha=0.4,
                color="black",
            )
            axs[1].plot(
                [np.min(self.prediction_test), np.max(self.prediction_test)],
                [np.min(self.prediction_test), np.max(self.prediction_test)],
                color="red",
                alpha=0.5,
            )
            axs[2].hist(
                self.testing_set.future
                - self.prediction_test.reshape(self.prediction_test.shape[0],),
                bins=20,
                alpha=0.7,
                color="black",
            )
        else:
            plt.close()
            fig, axs = plt.subplots(3)
            axs[0].plot(
                self.training_set.DATE,
                self.training_set.future,
                label="Realized Volatility",
                alpha=0.5,
                color="black",
                lw=0.5,
            )
            axs[0].plot(
                self.training_set.DATE,
                self.prediction_train,
                label="Prediction",
                alpha=0.8,
                lw=1,
            )
            axs[0].legend()
            axs[1].plot(
                self.training_set.future,
                self.prediction_train,
                "o",
                alpha=0.4,
                color="black",
            )
            axs[1].plot(
                [np.min(self.prediction_train), np.max(self.prediction_train)],
                [np.min(self.prediction_train), np.max(self.prediction_train)],
                color="red",
                alpha=0.5,
            )
            axs[2].hist(
                self.training_set.future
                - self.prediction_train.reshape(self.prediction_train.shape[0],),
                bins=50,
                alpha=0.7,
                color="black",
            )
