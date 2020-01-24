from run_HAR_model import load_data
from LSTM import *

df_input = load_data()

lstm_instance = DataPreparationLSTM(
    df=df_input,
    future=1,
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

x = TrainLSTM(
    lstm_instance.training_set,
    lstm_instance.testing_set,
    epochs=20,
    learning_rate=0.005,
    layer_one=20,
    layer_two=10,
    layer_three=0,
    layer_four=0,
)
x.make_accuracy_measures()

x.make_performance_plot(show_testing_sample=True)
