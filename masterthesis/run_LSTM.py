from run_HAR_model import load_data, results
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
lstm_instance.future_values
lstm_instance.historical_values
lstm_instance.df_processed_data

tf.keras.backend.clear_session()  # clear session
x = TrainLSTM(
    lstm_instance.training_set,
    lstm_instance.testing_set,
    epochs=10,
    learning_rate=0.1,
    layer_one=2,
    layer_two=20,
    layer_three=0,
    layer_four=0,
)
x.train_lstm()
x.make_accuracy_measures()
x.fitness

x.make_performance_plot(show_testing_sample=False)


plt.close()
plt.plot(
    lstm_instance.training_set.DATE,
    lstm_instance.back_transformation(x.prediction_train),
    label="Prediction LSTM",
    lw=0.5,
)
plt.plot(
    lstm_instance.training_set.DATE,
    lstm_instance.back_transformation(np.array(x.training_set.future).reshape(-1, 1)),
    label="Realized Volatility",
    lw=1,
)
plt.plot(
    results["har_1_True"].training_set.DATE,
    results["har_1_True"].prediction_train,
    label="Prediction HAR",
    lw=0.5,
)
plt.plot(
    results["har_1_True"].training_set.DATE,
    results["har_1_True"].training_set.future,
    label="Realized Volatility",
    lw=1,
)
plt.legend()


plt.close()
plt.plot(
    lstm_instance.back_transformation(np.array(x.training_set.future).reshape(-1, 1)),
    lstm_instance.back_transformation(x.prediction_train),
    "o",
    color="green",
    alpha=0.2,
    label="LSTM Prediction",
)
plt.plot(
    results["har_1_True"].training_set.future,
    results["har_1_True"].prediction_train,
    "o",
    color="black",
    alpha=0.2,
    label="HAR Prediction",
)
plt.legend()
