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
        self.models_all = None
        self.lstm_data_dictionary = dict()
        self.har_data_dictionary = dict()
        self.all_predictions_dic = dict()
        self.accuracy_measure = dict()
        self.accuracy_measure_data_frame = None

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

        self.models_all = {
            "LSTM_SV": model_lstm_sv,
            "LSTM_RV": model_lstm_rv,
            "HAR_SV": model_har_sv,
            "HAR_RV": model_har_rv,
        }

    def prepare_lstm_data(self):

        if self.data_train_test is None:
            self.load_data()

        if self.models_all is None:
            self.load_saved_models()

        data_frame_map = {
            "train_test": self.data_train_test,
            "validation": self.data_validation,
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

        if self.models_all is None:
            self.load_saved_models()

        data_frame_map = {
            "train_test": self.data_train_test,
            "validation": self.data_validation,
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

    def make_prediction(self):
        mapping_help = {
            "LSTM_RV": ["train_test_False", "validation_False"],
            "LSTM_SV": ["train_test_True", "validation_True"],
            "HAR_RV": ["train_test_False", "validation_False"],
            "HAR_SV": ["train_test_True", "validation_True"],
        }

        for i_model in mapping_help.keys():
            selected_model = self.models_all[i_model]

            if i_model in ["HAR_SV", "HAR_RV"]:

                if i_model == "HAR_SV":
                    for i_data_set in mapping_help[i_model]:
                        data = self.har_data_dictionary[i_data_set]

                        prediction_tmp = selected_model.model.predict(
                            data.training_set[["RSV_plus", "RSV_minus", "RV_w", "RV_m"]]
                        )

                        self.all_predictions_dic[
                            "{}_{}_train".format(i_model, i_data_set)
                        ] = prediction_tmp

                        self.accuracy_measure[
                            "{}_{}_train".format(i_model, i_data_set)
                        ] = [
                            metrics.mean_absolute_error(
                                data.training_set.future, prediction_tmp
                            ),
                            metrics.mean_squared_error(
                                data.training_set.future, prediction_tmp
                            ),
                            metrics.r2_score(data.training_set.future, prediction_tmp),
                        ]

                        prediction_tmp = selected_model.model.predict(
                            data.testing_set[["RSV_plus", "RSV_minus", "RV_w", "RV_m"]]
                        )
                        self.all_predictions_dic[
                            "{}_{}_test".format(i_model, i_data_set)
                        ] = prediction_tmp

                        self.accuracy_measure[
                            "{}_{}_test".format(i_model, i_data_set)
                        ] = [
                            metrics.mean_absolute_error(
                                data.testing_set.future, prediction_tmp
                            ),
                            metrics.mean_squared_error(
                                data.testing_set.future, prediction_tmp
                            ),
                            metrics.r2_score(data.testing_set.future, prediction_tmp),
                        ]

                else:
                    for i_data_set in mapping_help[i_model]:
                        data = self.har_data_dictionary[i_data_set]

                        prediction_tmp = selected_model.model.predict(
                            data.training_set[["RV_t", "RV_w", "RV_m"]]
                        )
                        self.all_predictions_dic[
                            "{}_{}_train".format(i_model, i_data_set)
                        ] = prediction_tmp
                        self.accuracy_measure[
                            "{}_{}_train".format(i_model, i_data_set)
                        ] = [
                            metrics.mean_absolute_error(
                                data.training_set.future, prediction_tmp
                            ),
                            metrics.mean_squared_error(
                                data.training_set.future, prediction_tmp
                            ),
                            metrics.r2_score(data.training_set.future, prediction_tmp),
                        ]

                        prediction_tmp = selected_model.model.predict(
                            data.testing_set[["RV_t", "RV_w", "RV_m"]]
                        )
                        self.all_predictions_dic[
                            "{}_{}_test".format(i_model, i_data_set)
                        ] = prediction_tmp
                        self.accuracy_measure[
                            "{}_{}_test".format(i_model, i_data_set)
                        ] = [
                            metrics.mean_absolute_error(
                                data.testing_set.future, prediction_tmp
                            ),
                            metrics.mean_squared_error(
                                data.testing_set.future, prediction_tmp
                            ),
                            metrics.r2_score(data.testing_set.future, prediction_tmp),
                        ]

            else:
                for i_data_set in mapping_help[i_model]:
                    data = self.lstm_data_dictionary[i_data_set]

                    prediction_tmp = selected_model.predict(data.train_matrix)
                    self.all_predictions_dic[
                        "{}_{}_train".format(i_model, i_data_set)
                    ] = data.back_transformation(prediction_tmp)

                    self.accuracy_measure["{}_{}_train".format(i_model, i_data_set)] = [
                        metrics.mean_absolute_error(
                            data.back_transformation(
                                np.array(data.training_set.future).reshape(-1, 1)
                            ),
                            data.back_transformation(prediction_tmp),
                        ),
                        metrics.mean_squared_error(
                            data.back_transformation(
                                np.array(data.training_set.future).reshape(-1, 1)
                            ),
                            data.back_transformation(prediction_tmp),
                        ),
                        metrics.r2_score(
                            data.back_transformation(
                                np.array(data.training_set.future).reshape(-1, 1)
                            ),
                            data.back_transformation(prediction_tmp),
                        ),
                    ]

                    prediction_tmp = selected_model.predict(data.test_matrix)
                    self.all_predictions_dic[
                        "{}_{}_test".format(i_model, i_data_set)
                    ] = data.back_transformation(prediction_tmp)

                    self.accuracy_measure["{}_{}_test".format(i_model, i_data_set)] = [
                        metrics.mean_absolute_error(
                            data.back_transformation(
                                np.array(data.testing_set.future).reshape(-1, 1)
                            ),
                            data.back_transformation(prediction_tmp),
                        ),
                        metrics.mean_squared_error(
                            data.back_transformation(
                                np.array(data.testing_set.future).reshape(-1, 1)
                            ),
                            data.back_transformation(prediction_tmp),
                        ),
                        metrics.r2_score(
                            data.back_transformation(
                                np.array(data.testing_set.future).reshape(-1, 1)
                            ),
                            data.back_transformation(prediction_tmp),
                        ),
                    ]

        df = pd.DataFrame.from_dict(self.accuracy_measure)
        df = df.transpose()
        df.columns = ["MAE", "MSE", "RSquared"]

        df_ = df.copy()
        df_.reset_index(level=0, inplace=True)
        df_.rename(columns={"index": "Model"}, inplace=True)

        self.accuracy_measure_data_frame = df_


result_instance = ResultOutput(forecast_period=20)
result_instance.prepare_lstm_data()
result_instance.prepare_har_data()
result_instance.make_prediction()

df = result_instance.accuracy_measure_data_frame.copy()
df["RV"] = df.apply(lambda x: "RV" in x.Model, axis=1)
df_RV = df[df.RV == True].drop(columns="RV")
df_SV = df[df.RV == False].drop(columns="RV")

print(df_RV.to_latex())
print(
    df_SV.to_latex()
)  # export it to text document in this format (similar to estimation results of HAR model)
