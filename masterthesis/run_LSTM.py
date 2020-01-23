from run_HAR_model import load_data
from LSTM import *
import matplotlib.pyplot as plt

df_input = load_data()

lstm_instance = DataPreparationLSTM(
    df=df_input,
    future=5,
    standard_scaler=False,
    min_max_scaler=True,
    log_transform=True,
    semi_variance=True,
)
lstm_instance.prepare_complete_data_set()

x = TrainLSTM(lstm_instance.training_set, lstm_instance.testing_set, epochs=20)
x.make_accuracy_measures()


prediction = x.fitted_model.predict(x.train_matrix)

# terrific little graph
plt.close()
fig, axs = plt.subplots(3)
axs[0].plot(
    x.training_set.DATE,
    x.training_set.future,
    label="Realized Volatility",
    alpha=0.5,
    color="black",
)
axs[0].plot(x.training_set.DATE, prediction, label="Prediction", alpha=0.8)
axs[0].legend()
axs[1].plot(x.training_set.future, prediction, "o", alpha=0.4, color="black")
axs[1].plot(
    [np.min(prediction), np.max(prediction)],
    [np.min(prediction), np.max(prediction)],
    color="red",
    alpha=0.5,
)
axs[2].hist(
    x.training_set.future - prediction.reshape(prediction.shape[0],),
    bins=50,
    alpha=0.7,
    color="black",
)
