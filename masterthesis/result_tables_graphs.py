from run_HAR_model import results
from HAR_Model import *
from LSTM import *
from config import *


class ResultOutput:
    def __init__(
        self, forecast_period: int = 1,
    ):
        self.forecast_period = forecast_period

        self.data_train_test = None
        self.data_validation = None
        self.model_lstm = None
        self.model_har = None
        self.lstm_data_dictionary = dict()
        self.har_data_dictionary = dict()

    def load_data(self):
        df_m = pd.read_csv(
            instance_path.path_input + "/" + "RealizedMeasures03_10.csv", index_col=0
        )
        df_m.DATE = df_m.DATE.values
        df_m.DATE = pd.to_datetime(df_m.DATE, format="%Y%m%d")
        self.data_train_test = df_m

        df = pd.read_csv(
            instance_path.path_input + "/" + "DataFeatures.csv", index_col=0
        )
        df.DATE = df.DATE.values
        df.DATE = pd.to_datetime(df.DATE, format="%Y%m%d")
        self.data_validation = df

    def load_saved_models(self):

        model_lstm_sv = tf.keras.models.load_model(
            "LSTM_SV_{}.h5".format(self.forecast_period)
        )  # HARD CODED NAMES !!!
        model_lstm_rv = tf.keras.models.load_model(
            "LSTM_RV_{}.h5".format(self.forecast_period)
        )  # this model is not saved yet (hyperparameter optimization)

        model_har_sv = results["har_{}_True".format(self.forecast_period)]
        model_har_rv = results["har_{}_False".format(self.forecast_period)]

        self.model_lstm = {"LSTM_SV": model_lstm_sv, "LSTM_RV": model_lstm_rv}
        self.model_har = {"HAR_SV": model_har_sv, "HAR_RV": model_har_rv}

    def prepare_lstm_data(self):

        if self.data_train_test is None:
            self.load_data()

        if self.model_har is None:
            self.load_saved_models()

        data_frame_map = {
            "train_test": self.data_train_test,
            "validation": self.data_train_test,
        }

        for data_frame in data_frame_map.keys():

            if data_frame_map[data_frame].equals(self.data_validation):
                period_train_ = list(
                    [
                        pd.to_datetime("20110103", format="%Y%m%d"),
                        pd.to_datetime("20111231", format="%Y%m%d"),
                    ]
                )
                period_test_ = list(
                    [
                        pd.to_datetime("20110103", format="%Y%m%d"),
                        pd.to_datetime("20111231", format="%Y%m%d"),
                    ]
                )
            else:
                period_train_ = list(
                    [
                        pd.to_datetime("20030910", format="%Y%m%d"),
                        pd.to_datetime("20091231", format="%Y%m%d"),
                    ]
                )
                period_test_ = list(
                    [
                        pd.to_datetime("20100101", format="%Y%m%d"),
                        pd.to_datetime("20101231", format="%Y%m%d"),
                    ]
                )

            for semi_variance_indication in [True, False]:

                # LSTM instance initiation
                lstm_instance = DataPreparationLSTM(
                    df=data_frame_map[data_frame],
                    future=self.forecast_period,
                    lag=20,
                    standard_scaler=False,
                    min_max_scaler=True,
                    log_transform=True,
                    semi_variance=semi_variance_indication,
                    jump_detect=True,
                    period_train=period_train_,
                    period_test=period_test_,
                )

                lstm_instance.prepare_complete_data_set()
                lstm_instance.reshape_input_data()

                self.lstm_data_dictionary[
                    "{}_{}".format(data_frame, semi_variance_indication)
                ] = lstm_instance

    def prepare_har_data(self):

        if self.data_train_test is None:
            self.load_data()

        if self.model_har is None:
            self.load_saved_models()

        data_frame_map = {
            "train_test": self.data_train_test,
            "validation": self.data_train_test,
        }

        for data_frame in data_frame_map.keys():

            if data_frame_map[data_frame].equals(self.data_validation):
                period_train_ = list(
                    [
                        pd.to_datetime("20110103", format="%Y%m%d"),
                        pd.to_datetime("20111231", format="%Y%m%d"),
                    ]
                )
                period_test_ = list(
                    [
                        pd.to_datetime("20110103", format="%Y%m%d"),
                        pd.to_datetime("20111231", format="%Y%m%d"),
                    ]
                )
            else:
                period_train_ = list(
                    [
                        pd.to_datetime("20030910", format="%Y%m%d"),
                        pd.to_datetime("20091231", format="%Y%m%d"),
                    ]
                )
                period_test_ = list(
                    [
                        pd.to_datetime("20100101", format="%Y%m%d"),
                        pd.to_datetime("20101231", format="%Y%m%d"),
                    ]
                )

            for semi_variance_indication in [True, False]:
                # LSTM instance initiation
                har_instance = HARModel(
                    df=data_frame_map[data_frame],
                    future=self.forecast_period,
                    lags=[4, 20],
                    feature="RV",
                    semi_variance=semi_variance_indication,
                    jump_detect=True,
                    period_train=period_train_,
                    period_test=period_test_,
                )

                har_instance.make_testing_training_set()

                self.har_data_dictionary[
                    "{}_{}".format(data_frame, semi_variance_indication)
                ] = har_instance

    # def make_prediction(self):


x = ResultOutput(forecast_period=1)

x.load_data()
x.load_saved_models()
x.prepare_lstm_data()
x.prepare_har_data()

list(x.lstm_data_dictionary.keys())[0]  # accessing a given data set
