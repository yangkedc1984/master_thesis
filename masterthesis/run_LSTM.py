from run_HAR_model import load_data, results
from LSTM import *
from config import *

df_input = load_data()

lstm_instance = DataPreparationLSTM(
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
lstm_instance.reshape_input_data()

tf.keras.backend.clear_session()
x = TrainLSTM(
    lstm_instance.training_set,
    lstm_instance.testing_set,
    epochs=40,
    learning_rate=0.05,
    layer_one=20,
    layer_two=20,
    layer_three=0,
    layer_four=0,
)
x.make_accuracy_measures()

# x.make_performance_plot(show_testing_sample=False)

# x.fitted_model.save(folder_structure.output_LSTM + "/" + "LSTM_SV_1.h5")
# x.fitted_model.save(folder_structure.output_LSTM + "/" + "LSTM_RV_1.h5")
# x.fitted_model.save(folder_structure.output_LSTM + "/" + "LSTM_SV_5.h5")
# x.fitted_model.save(folder_structure.output_LSTM + "/" + "LSTM_RV_5.h5")
# x.fitted_model.save(folder_structure.output_LSTM + "/" + "LSTM_SV_20.h5")
# x.fitted_model.save(folder_structure.output_LSTM + "/" + "LSTM_RV_20.h5")
