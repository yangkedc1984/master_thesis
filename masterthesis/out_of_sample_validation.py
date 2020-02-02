from run_LSTM import *  # runs the LSTM model defined in the run_LSTM.py module
from config import *

df = pd.read_csv(instance_path.path_input + "/" + "DataFeatures.csv", index_col=0)
df.DATE = df.DATE.values
df.DATE = pd.to_datetime(df.DATE, format="%Y%m%d")

lstm_validation_data = DataPreparationLSTM(
    df=df,
    future=20,
    lag=20,
    standard_scaler=False,
    min_max_scaler=True,
    log_transform=True,
    semi_variance=True,
    jump_detect=True,
    period_train=list(
        [
            pd.to_datetime("20110103", format="%Y%m%d"),
            pd.to_datetime("20111231", format="%Y%m%d"),
        ]
    ),
    period_test=list(
        [
            pd.to_datetime("20110103", format="%Y%m%d"),
            pd.to_datetime("20111231", format="%Y%m%d"),
        ]
    ),
)
lstm_validation_data.prepare_complete_data_set()
lstm_validation_data.reshape_input_data()


plt.close()
fig, axs = plt.subplots(3)
axs[0].plot(
    lstm_validation_data.training_set.DATE,
    lstm_validation_data.training_set.future,
    label="Realized Volatility",
    alpha=0.5,
    color="black",
    lw=0.5,
)
axs[0].plot(
    lstm_validation_data.training_set.DATE,
    x.fitted_model.predict(lstm_validation_data.train_matrix),
    label="Prediction",
    alpha=0.8,
    lw=1,
)
axs[0].legend()
axs[1].plot(
    lstm_validation_data.training_set.future,
    x.fitted_model.predict(lstm_validation_data.train_matrix),
    "o",
    alpha=0.4,
    color="black",
)
axs[1].plot(
    [
        np.min(x.fitted_model.predict(lstm_validation_data.train_matrix)),
        np.max(x.fitted_model.predict(lstm_validation_data.train_matrix)),
    ],
    [
        np.min(x.fitted_model.predict(lstm_validation_data.train_matrix)),
        np.max(x.fitted_model.predict(lstm_validation_data.train_matrix)),
    ],
    color="red",
    alpha=0.5,
)
axs[2].hist(
    lstm_validation_data.training_set.future
    - x.fitted_model.predict(lstm_validation_data.train_matrix).reshape(
        lstm_validation_data.train_y.shape[0],
    ),
    bins=20,
    alpha=0.7,
    color="black",
)

x.make_performance_plot(show_testing_sample=False)
