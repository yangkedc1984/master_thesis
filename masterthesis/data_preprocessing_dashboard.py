# Preprocessing Data for the dashboard
# goal is to get a data frame with all the predictions and the data

from run_AutoRegression_Model import *
from config import folder_structure
from LSTM import TimeSeriesDataPreparationLSTM


df_m = pd.read_csv(
    folder_structure.path_input + "/" + "RealizedMeasures03_10.csv", index_col=0
)
df_m.DATE = df_m.DATE.values
df_m.DATE = pd.to_datetime(df_m.DATE, format="%Y%m%d")
TimeSeriesDataPreparationLSTM(df_m)

df = pd.read_csv(folder_structure.path_input + "/" + "DataFeatures.csv", index_col=0)
df.DATE = df.DATE.values
df.DATE = pd.to_datetime(df.DATE, format="%Y%m%d")


# LSTM models: 'LSTM_SV_1_hist20' or LSTM_SV_1_hist40'
model_lstm_sv = tf.keras.models.load_model(
    folder_structure.output_LSTM + "/" + "LSTM_SV_{}.h5".format(1)
)

dict_lstm = {}
for i in ["SV", "RV"]:
    for j in [1, 5]:  # 20
        for k in ["hist40", "hist20"]:
            dict_lstm["LSTM_{}_{}_{}".format(i, j, k)] = tf.keras.models.load_model(
                folder_structure.output_LSTM + "/" + "LSTM_{}_{}_{}.h5".format(i, j, k)
            )
dict_lstm

semi_variance_list = [True, False]
periods_list = [1, 5, 20]
lags_list = [20, 40]

for semi_variance in range(2):
    for period in range(3):
        for lags in range(2):
            lstm_model = tf.keras.models.load_model(
                folder_structure.output_LSTM
                + "/"
                + "LSTM_{}_{}_{}.h5".format(
                    semi_variance_list[semi_variance],
                    periods_list[period],
                    lags_list[lags],
                )
            )
            lstm_model

            lstm_instance = TimeSeriesDataPreparationLSTM(
                df=df_m,  # has to be the overall data set !!!
                future=periods_list[period],
                lag=lags_list[lags],
                standard_scaler=False,
                min_max_scaler=True,
                log_transform=True,
                semi_variance=semi_variance_list[semi_variance],
                jump_detect=True,
                period_train=list(
                    [
                        pd.to_datetime("20030910", format="%Y%m%d"),
                        pd.to_datetime("20111231", format="%Y%m%d"),
                    ]
                ),
                period_test=list(
                    [
                        pd.to_datetime("20030910", format="%Y%m%d"),
                        pd.to_datetime("20030910", format="%Y%m%d"),
                    ]
                ),
            )
            lstm_instance.prepare_complete_data_set()

            prediction_tmp = lstm_instance.back_transformation(
                lstm_model.predict(lstm_instance.train_matrix)
            )


lstm_instance = TimeSeriesDataPreparationLSTM(
    df=df_m,
    future=20,  # 1, 5, 20
    lag=40,  # 20 40
    standard_scaler=False,
    min_max_scaler=True,
    log_transform=True,
    semi_variance=False,  # True False
    jump_detect=True,
    period_train=list(
        [
            pd.to_datetime("20030910", format="%Y%m%d"),
            pd.to_datetime("20111231", format="%Y%m%d"),
        ]
    ),
    period_test=list(
        [
            pd.to_datetime("20030910", format="%Y%m%d"),
            pd.to_datetime("20111231", format="%Y%m%d"),
        ]
    ),
)
lstm_instance.prepare_complete_data_set()
lstm_instance.reshape_input_data()

prediction_tmp = lstm_model.predict(lstm_instance.train_matrix)

prediction_tmp = lstm_instance.back_transformation(prediction_tmp)

plt.close()
plt.plot(prediction_tmp)

# HAR Model Data Transformation
df_m = df_m[["DATE", "RV", "RSV_minus", "RSV_plus"]]
data_frame_final = pd.concat([pd.DataFrame(df_m), pd.DataFrame(df)]).reset_index(
    drop=True
)


def make_exponential(series, log_transformed: bool = False):
    if log_transformed:
        return np.exp(series)
    else:
        return series


future_periods = [1, 5, 20]
semi_variance_list = [True, False]
log_trans_list = [True, False]
df_final = pd.DataFrame()
df_export = pd.DataFrame()
df_export_2 = pd.DataFrame()

for log_trans in range(2):
    for semi_variance in range(2):
        for i in range(3):
            har_instance = HARModelLogTransformed(
                df=data_frame_final,
                future=future_periods[i],
                lags=[4, 20],
                feature="RV",
                semi_variance=semi_variance_list[semi_variance],
                jump_detect=True,
                log_transformation=log_trans_list[log_trans],
                period_train=list(
                    [
                        pd.to_datetime("20030910", format="%Y%m%d"),
                        pd.to_datetime("20111231", format="%Y%m%d"),
                    ]
                ),
                period_test=list(
                    [
                        pd.to_datetime("20030910", format="%Y%m%d"),
                        pd.to_datetime("20030910", format="%Y%m%d"),
                    ]
                ),
            )
            har_instance.make_testing_training_set()

            if semi_variance_list[semi_variance] is True:
                har_variables = ["RSV_plus", "RSV_minus", "RV_w", "RV_m"]
                semi_variance_indication = "SV"
            else:
                har_variables = ["RV_t", "RV_w", "RV_m"]
                semi_variance_indication = "RV"

            if log_trans_list[log_trans] is True:
                log_trans_indication = ",L"
            else:
                log_trans_indication = ""

            df_tmp = pd.DataFrame(
                results_har[
                    "har_{}_{}_{}".format(
                        future_periods[i],
                        semi_variance_list[semi_variance],
                        log_trans_list[log_trans],
                    )
                ].model.predict(har_instance.training_set[har_variables]),
                columns={
                    "H({}{})".format(semi_variance_indication, log_trans_indication)
                },
            ).reset_index()
            df_tmp["period"] = future_periods[i]

            df_tmp[
                "H({}{})".format(semi_variance_indication, log_trans_indication)
            ] = make_exponential(
                df_tmp[
                    "H({}{})".format(semi_variance_indication, log_trans_indication)
                ],
                log_transformed=log_trans_list[log_trans],
            )

            df_x = pd.DataFrame(
                har_instance.training_set.DATE, columns={"DATE"}
            ).reset_index()
            df_x = df_x.merge(df_tmp, on="index")
            df_x = df_x.drop(columns=["index"])

            if i == 0:
                df_final = df_x
            else:
                df_final = pd.concat([df_final, df_x])

        if semi_variance == 0:
            df_export = df_final
        else:
            df_export = df_export.merge(df_final, on=["period", "DATE"])

    if log_trans == 0:
        df_export_2 = df_export
    else:
        df_export_2 = df_export_2.merge(df_export, on=["period", "DATE"])

# add this shit to a loop
df_export_2["dataset"] = np.NAN

df_export_2.dataset[
    (df_export_2.DATE >= pd.to_datetime("20040910", format="%Y%m%d"))
    & (df_export_2.DATE < pd.to_datetime("20050910", format="%Y%m%d"))
] = "training"

df_export_2.dataset[
    (df_export_2.DATE >= pd.to_datetime("20040910", format="%Y%m%d"))
    & (df_export_2.DATE < pd.to_datetime("20050910", format="%Y%m%d"))
] = "testing"

df_export_2.dataset[
    (df_export_2.DATE >= pd.to_datetime("20040910", format="%Y%m%d"))
    & (df_export_2.DATE < pd.to_datetime("20050910", format="%Y%m%d"))
] = "validation"

df_export_2 = df_export_2.merge(data_frame_final[["DATE", "RV"]], on="DATE")  # works
