from run_HAR_model import *
from LSTM import *

df_input = load_data()

lstm_instance = TimeSeriesDataPreparationLSTM(
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

tf.keras.backend.clear_session()
x = TrainLSTM(
    lstm_instance.training_set,
    lstm_instance.testing_set,
    epochs=20,
    learning_rate=0.01,
    layer_one=40,
    layer_two=37,
    layer_three=6,
    layer_four=0,
    adam_optimizer=True,
)
x.make_accuracy_measures()

x.fitness
x.train_accuracy
x.test_accuracy

x.make_performance_plot(show_testing_sample=False)

# {'MSE': 0.004486546327296942, 'MAE': 0.051842309968481576, 'RSquared': 0.8052563696263175}  :: train accuracy
# {'MSE': 0.00580020601640592, 'MAE': 0.05804518833227246, 'RSquared': 0.5745797767210921} :: test accuracy
# 0.00125 45.0 3.0 15.0 20.0  (from first Genetic algorithm)

# x.fitted_model.save(folder_structure.output_LSTM + "/" + "LSTM_hist40_sv_1_aftergeneticalgorithm.h5")
# x.fitted_model.save(folder_structure.output_LSTM + "/" + "new_run_model.h5")
# x.fitted_model.save(folder_structure.output_LSTM + "/" + "LSTM_SV_1.h5")
# x.fitted_model.save(folder_structure.output_LSTM + "/" + "LSTM_RV_1.h5")
# x.fitted_model.save(folder_structure.output_LSTM + "/" + "LSTM_SV_5.h5")
# x.fitted_model.save(folder_structure.output_LSTM + "/" + "LSTM_RV_5.h5")
# x.fitted_model.save(folder_structure.output_LSTM + "/" + "LSTM_SV_20.h5")
# x.fitted_model.save(folder_structure.output_LSTM + "/" + "LSTM_RV_20.h5")
