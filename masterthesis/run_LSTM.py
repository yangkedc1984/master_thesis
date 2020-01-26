from run_HAR_model import load_data
from LSTM import *

df_input = load_data()

lstm_instance = DataPreparationLSTM(
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

tf.keras.backend.clear_session()  # important to clear session!
x = TrainLSTM(
    lstm_instance.training_set,
    lstm_instance.testing_set,
    epochs=10,
    learning_rate=0.005,
    layer_one=10,
    layer_two=10,
    layer_three=5,
    layer_four=5,
)
x.make_accuracy_measures()
x.fitness


# x.make_performance_plot(show_testing_sample=True)
#
#
# # back transformation:
# plt.close()
# plt.plot(
#     np.exp(lstm_instance.applied_scaler.inverse_transform(x.prediction_test)),
#     np.exp(
#         lstm_instance.applied_scaler.inverse_transform(
#             np.array(x.testing_set.future).reshape(-1, 1)
#         )
#     ),
#     "o",
#     alpha=0.25,
#     color="black",
# )
