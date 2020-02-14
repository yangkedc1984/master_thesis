from run_HAR_model import *
from LSTM import *

df_input = load_data()
df_input.RV = np.log(df_input.RV)
df_input.RSV_minus = np.log(df_input.RSV_minus)

lstm_instance = TimeSeriesDataPreparationLSTM(
    df=df_input,
    future=5,
    lag=40,
    standard_scaler=False,
    min_max_scaler=False,
    log_transform=False,
    semi_variance=True,
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
lstm_instance.training_set


tf.keras.backend.clear_session()
x = TrainLSTM(
    lstm_instance.training_set,
    lstm_instance.testing_set,
    epochs=10,
    learning_rate=0.001,
    layer_one=40,
    layer_two=20,
    layer_three=10,
    layer_four=2,
)
# x.make_accuracy_measures()

# x.train_accuracy
# x.test_accuracy

# x.make_performance_plot(show_testing_sample=False)


# print(
#    x.test_accuracy, x.train_accuracy
# )  # this accuracy measure is based on scaled data!!!  (got a massively better R squared)


# x.fitted_model.save(folder_structure.output_LSTM + "/" + "new_run_model.h5")
# x.fitted_model.save(folder_structure.output_LSTM + "/" + "LSTM_SV_1.h5")
# x.fitted_model.save(folder_structure.output_LSTM + "/" + "LSTM_RV_1.h5")
# x.fitted_model.save(folder_structure.output_LSTM + "/" + "LSTM_SV_5.h5")
# x.fitted_model.save(folder_structure.output_LSTM + "/" + "LSTM_RV_5.h5")
# x.fitted_model.save(folder_structure.output_LSTM + "/" + "LSTM_SV_20.h5")
# x.fitted_model.save(folder_structure.output_LSTM + "/" + "LSTM_RV_20.h5")
