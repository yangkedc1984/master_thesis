from run_HAR_model import load_data
from config import folder_structure
from LSTM import *

df_input = load_data()

lstm_instance = TimeSeriesDataPreparationLSTM(
    df=df_input,
    future=20,
    lag=20,
    standard_scaler=False,
    min_max_scaler=True,
    log_transform=True,
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


tf.keras.backend.clear_session()
best_model = TrainLSTM(
    lstm_instance.training_set,
    lstm_instance.testing_set,
    epochs=80,
    learning_rate=0.01,
    layer_one=40,
    layer_two=40,
    layer_three=0,
    layer_four=0,
    adam_optimizer=True,
)
best_model.make_accuracy_measures()

# best_model.fitted_model.save(
#     folder_structure.output_LSTM + "/" + "LSTM_True_20_20_v22.h5"
# )
