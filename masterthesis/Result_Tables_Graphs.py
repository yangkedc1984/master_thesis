from run_HAR_model import results_har
from HAR_Model import *
from LSTM import *
from config import folder_structure, os, time


class ResultOutput:
    def __init__(self, forecast_period: int = 1, log_transformation: bool = False):
        self.forecast_period = forecast_period
        self.log_transformation = log_transformation

        self.data_train_test = None
        self.data_validation = None
        self.models_all = None
        self.lstm_data_dictionary = dict()
        self.har_data_dictionary = dict()
        self.all_predictions_dic = dict()
        self.accuracy_measure = dict()
        self.accuracy_measure_data_frame = None
        self.data_frame_for_plot = None
        self.unit_test_lstm = None
        self.unit_test_har = None

    def load_data(self):
        df_m = pd.read_csv(
            folder_structure.path_input + "/" + "RealizedMeasures03_10.csv", index_col=0
        )
        df_m.DATE = df_m.DATE.values
        df_m.DATE = pd.to_datetime(df_m.DATE, format="%Y%m%d")
        self.data_train_test = df_m

        df = pd.read_csv(
            folder_structure.path_input + "/" + "DataFeatures.csv", index_col=0
        )
        df.DATE = df.DATE.values
        df.DATE = pd.to_datetime(df.DATE, format="%Y%m%d")
        self.data_validation = df

    def load_saved_models(self):

        model_lstm_sv = tf.keras.models.load_model(
            folder_structure.output_LSTM
            + "/"
            + "LSTM_SV_{}.h5".format(self.forecast_period)
        )

        model_lstm_rv = tf.keras.models.load_model(
            folder_structure.output_LSTM
            + "/"
            + "LSTM_RV_{}.h5".format(self.forecast_period)
        )

        model_har_sv = results_har[
            "har_{}_True_{}".format(self.forecast_period, self.log_transformation)
        ]
        model_har_rv = results_har[
            "har_{}_False_{}".format(self.forecast_period, self.log_transformation)
        ]

        self.models_all = {
            "LSTM_SV": model_lstm_sv,
            "LSTM_RV": model_lstm_rv,
            "HAR_SV": model_har_sv,
            "HAR_RV": model_har_rv,
        }  # should we also load the AR(1) & AR(3) models?  # we should load har log models aswell!!!

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
                lstm_instance = TimeSeriesDataPreparationLSTM(
                    df=data_frame_map[data_frame],
                    future=self.forecast_period,
                    lag=20,  # have to be changed
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

        self.unit_test_lstm = self.lstm_data_dictionary["train_test_True"]  # added

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
                har_instance = HARModelLogTransformed(
                    df=data_frame_map[data_frame],
                    future=self.forecast_period,
                    lags=[4, 20],
                    feature="RV",
                    semi_variance=semi_variance_indication,
                    jump_detect=True,
                    log_transformation=self.log_transformation,
                    period_train=period_train_,
                    period_test=period_test_,
                )

                har_instance.make_testing_training_set()

                self.har_data_dictionary[
                    "{}_{}".format(data_frame, semi_variance_indication)
                ] = har_instance

        self.unit_test_har = self.har_data_dictionary["train_test_True"]

    def make_prediction_accuracy_measurement(self):
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

    def export_result_tables(self, save: bool = True):
        df = self.accuracy_measure_data_frame.copy()
        df["RV"] = df.apply(lambda x: "RV" in x.Model, axis=1)
        df_rv = df[df.RV == True].drop(columns="RV")
        df_sv = df[df.RV == False].drop(columns="RV")

        if save:
            os.chdir(folder_structure.output_Tables)

            accuracy_measures_export_rv = open(
                "{}_RV_models_{}".format(time.ctime(), self.forecast_period), "a+"
            )
            accuracy_measures_export_rv.write(df_rv.to_latex())

            accuracy_measures_export_sv = open(
                "{}_SV_models_{}".format(time.ctime(), self.forecast_period), "a+"
            )
            accuracy_measures_export_sv.write(df_sv.to_latex())

    @staticmethod
    def make_plot(
        df_input,
        save_plot: bool = False,
        identifier_1="L",
        identifier_2="T",
        identifier_3="train",
        identifier_4=1,
    ):
        plt.close()
        fig3 = plt.figure(constrained_layout=True)
        gs = fig3.add_gridspec(3, 2)
        f3_ax1 = fig3.add_subplot(gs[0, :])
        f3_ax1.set_title("Time Series")
        f3_ax1.plot(
            df_input.DATE,
            df_input.LSTM_prediction,
            lw=1,
            color="darkgreen",
            label="LSTM Prediction",
        )
        f3_ax1.plot(
            df_input.DATE,
            df_input.HAR_prediction,
            lw=1,
            color="mediumseagreen",
            label="HAR Prediction",
        )
        f3_ax1.plot(
            df_input.DATE,
            df_input.future,
            lw=0.5,
            alpha=0.5,
            color="black",
            label="Realized Volatility",
        )
        f3_ax1.legend()
        f3_ax2 = fig3.add_subplot(gs[1, 0])
        f3_ax2.set_title("Mincer-Zarnowitz Regression")
        f3_ax2.plot(
            df_input.future,
            df_input.LSTM_prediction,
            "o",
            color="darkgreen",
            alpha=0.2,
            label="LSTM Prediction vs Realized Volatility",
        )
        f3_ax2.legend()
        f3_ax3 = fig3.add_subplot(gs[1, 1])
        f3_ax3.set_title("Mincer-Zarnowitz Regression")
        f3_ax3.plot(
            df_input.future,
            df_input.HAR_prediction,
            "o",
            color="mediumseagreen",
            alpha=0.2,
            label="LSTM Prediction vs Realized Volatility",
        )
        f3_ax3.legend()
        f3_ax4 = fig3.add_subplot(gs[2, 0])
        f3_ax4.set_title("Bias Histogram")
        f3_ax4.hist(
            df_input.future - df_input.LSTM_prediction,
            bins=50,
            color="darkgreen",
            alpha=0.7,
            label="LSTM Bias",
        )
        f3_ax4.hist(
            df_input.future - df_input.HAR_prediction,
            bins=50,
            color="mediumseagreen",
            alpha=0.4,
            label="HAR Bias",
        )
        f3_ax4.legend()
        f3_ax5 = fig3.add_subplot(gs[2, 1])
        f3_ax5.set_title("Bias Time Series")
        f3_ax5.plot(
            df_input.DATE,
            df_input.future - df_input.LSTM_prediction,
            lw=0.5,
            color="darkgreen",
            alpha=1,
            label="LSTM Bias",
        )
        f3_ax5.plot(
            df_input.DATE,
            df_input.future - df_input.HAR_prediction,
            lw=0.5,
            color="mediumseagreen",
            alpha=1,
            label="HAR Bias",
        )
        f3_ax5.legend()

        if save_plot:
            os.chdir(folder_structure.output_Graphs)
            fig3.savefig(
                "{}_{}_{}_{}.png".format(
                    identifier_1, identifier_2, identifier_3, identifier_4
                )
            )

    @staticmethod
    def make_plot_2(
        df,
        save_plot: bool = False,
        identifier_1="L",
        identifier_2="T",
        identifier_3="train",
        identifier_4=1,
    ):

        df["indicator"] = df.DATE.map(lambda x: 100 * x.year + x.month)

        for i in ["LSTM_prediction", "HAR_prediction", "future"]:
            df = df.merge(
                df.groupby("indicator")
                .apply(lambda x: x[i].mean())
                .to_frame(name="{}_average".format(i))
                .reset_index(level=0),
                on="indicator",
            )

        df["performance_check"] = np.where(
            np.abs(df.LSTM_prediction_average - df.future_average)
            > np.abs(df.HAR_prediction_average - df.future_average),
            "HAR Model",
            "LSTM",  # if == 1, then HAR Model is superior
        )
        models = ["LSTM_prediction_average", "HAR_prediction_average", "future_average"]
        colors = ["darkgreen", "mediumseagreen", "black"]
        plt.close()
        fig, axs = plt.subplots(2)
        for i in range(3):
            axs[0].plot(
                df.DATE, df[models[i]], label=models[i], alpha=1, lw=1, color=colors[i]
            )
        axs[0].legend()
        axs[1].plot(df.DATE, df.performance_check, alpha=0.3, lw=1, color="black")

        if save_plot:
            os.chdir(folder_structure.output_Graphs)
            fig.savefig(
                "Aggregated_{}_{}_{}_{}.png".format(
                    identifier_1, identifier_2, identifier_3, identifier_4
                )
            )

    def make_all_plot(self, save_all_plots: bool = False):

        mapping_data = {
            "RV": ["train_test_False", "validation_False"],
            "SV": ["train_test_True", "validation_True"],
        }
        mapping_graph = {"RV": ["LSTM_RV", "HAR_RV"], "SV": ["LSTM_SV", "HAR_SV"]}

        for i_model_type in mapping_graph.keys():  # for i in [RV, SV]:
            for i_data in mapping_data[
                i_model_type
            ]:  # for i in ["train_test_False", "validation_False"]:
                data_lstm = self.lstm_data_dictionary[i_data]  # self converts to self
                data_har = self.har_data_dictionary[i_data]

                selection_model_lstm = self.models_all[mapping_graph[i_model_type][0]]
                selection_model_har = self.models_all[mapping_graph[i_model_type][1]]

                df_lstm_help = data_lstm.training_set.DATE.to_frame(name="DATE")
                model_prediction = selection_model_lstm.predict(data_lstm.train_matrix)
                df_lstm_help["LSTM_prediction"] = data_lstm.back_transformation(
                    model_prediction
                )
                df_lstm_help["LSTM_future"] = data_lstm.back_transformation(
                    np.array(data_lstm.training_set.future).reshape(-1, 1)
                )

                df_har_help = data_har.training_set[["future", "DATE"]]

                if i_model_type == "RV":
                    df_har_help["HAR_prediction"] = selection_model_har.model.predict(
                        data_har.training_set[["RV_t", "RV_w", "RV_m"]]
                    )
                else:
                    df_har_help["HAR_prediction"] = selection_model_har.model.predict(
                        data_har.training_set[["RSV_plus", "RSV_minus", "RV_w", "RV_m"]]
                    )

                df_complete = df_lstm_help.merge(df_har_help, on="DATE")

                self.data_frame_for_plot = df_complete

                self.make_plot(
                    df_complete,
                    save_plot=save_all_plots,
                    identifier_1=i_model_type,
                    identifier_2=i_data,
                    identifier_3="train",
                    identifier_4=self.forecast_period,
                )

                self.make_plot_2(
                    df_complete,
                    save_plot=save_all_plots,
                    identifier_1=i_model_type,
                    identifier_2=i_data,
                    identifier_3="train",
                    identifier_4=self.forecast_period,
                )

                for i_model_type_test in mapping_graph.keys():
                    for i_data_test in mapping_data[i_model_type_test]:
                        data_lstm = self.lstm_data_dictionary[i_data_test]
                        data_har = self.har_data_dictionary[i_data_test]

                        selection_model_lstm = self.models_all[
                            mapping_graph[i_model_type_test][0]
                        ]
                        selection_model_har = self.models_all[
                            mapping_graph[i_model_type_test][1]
                        ]

                        df_lstm_help = data_lstm.testing_set.DATE.to_frame(name="DATE")
                        model_prediction = selection_model_lstm.predict(
                            data_lstm.test_matrix
                        )
                        df_lstm_help["LSTM_prediction"] = data_lstm.back_transformation(
                            model_prediction
                        )
                        df_lstm_help["LSTM_future"] = data_lstm.back_transformation(
                            np.array(data_lstm.testing_set.future).reshape(-1, 1)
                        )

                        df_har_help = data_har.testing_set[["future", "DATE"]]

                        if i_model_type_test == "RV":
                            df_har_help[
                                "HAR_prediction"
                            ] = selection_model_har.model.predict(
                                data_har.testing_set[["RV_t", "RV_w", "RV_m"]]
                            )
                        else:
                            df_har_help[
                                "HAR_prediction"
                            ] = selection_model_har.model.predict(
                                data_har.testing_set[
                                    ["RSV_plus", "RSV_minus", "RV_w", "RV_m"]
                                ]
                            )

                        df_complete = df_lstm_help.merge(df_har_help, on="DATE")

                        self.data_frame_for_plot = df_complete

                        self.make_plot(
                            df_complete,
                            save_plot=save_all_plots,
                            identifier_1=i_model_type_test,
                            identifier_2=i_data_test,
                            identifier_3="test",
                            identifier_4=self.forecast_period,
                        )

                        self.make_plot_2(
                            df_complete,
                            save_plot=save_all_plots,
                            identifier_1=i_model_type_test,
                            identifier_2=i_data_test,
                            identifier_3="test",
                            identifier_4=self.forecast_period,
                        )

    def run_all(self, save_: bool = True, save_plots: bool = True):
        self.prepare_lstm_data()
        self.prepare_har_data()

        # unit test: data pre processing
        assert (
            round(
                sum(
                    (
                        self.unit_test_lstm.back_transformation(
                            np.array(self.unit_test_lstm.testing_set.future).reshape(
                                -1, 1
                            )
                        ).reshape(self.unit_test_lstm.testing_set.shape[0],)
                        - self.unit_test_har.testing_set.future
                    )
                ),
                10,
            )
            == 0
        ), (
            "Error: Unequal future realized volatility for LSTM- and HAR Model"
            "(unequal dependent variable in the models)"
        )

        self.make_prediction_accuracy_measurement()
        self.export_result_tables(save=save_)
        self.make_all_plot(save_all_plots=save_plots)
