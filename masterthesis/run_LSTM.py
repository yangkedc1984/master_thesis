from run_HAR_model import load_data
from LSTM import *
import matplotlib.pyplot as plt

df_input = load_data()

lstm_instance = DataPreparationLSTM(
    df=df_input,
    future=1,
    standard_scaler=False,
    min_max_scaler=True,
    log_transform=True,
    semi_variance=True,
)

lstm_instance.prepare_complete_data_set()

x = TrainLSTM(lstm_instance.training_set, lstm_instance.testing_set)
x.reshape_input_data()
x.train_lstm_()


plt.plot(x.training_set.DATE, x.training_set.future)
plt.plot(x.training_set.DATE, x.fitted_model.predict(x.train_matrix))


# model which works!!!
model = tf.keras.models.Sequential()
model.add(
    tf.keras.layers.LSTM(
        16,
        activation=tf.nn.elu,
        input_shape=(x.train_matrix.shape[1], 1),
        return_sequences=True,
    )
)
model.add(tf.keras.layers.LSTM(8, activation=tf.nn.elu, return_sequences=True,))
model.add(tf.keras.layers.LSTM(8, activation=tf.nn.elu))
model.add(tf.keras.layers.Dense(1, activation="linear"))
model.compile(optimizer="adam", loss=tf.keras.losses.logcosh)

model.fit(x.train_matrix, x.train_y, epochs=5, verbose=1)

prediction = model.predict(x.train_matrix)

# terrific little graph
plt.close()
fig, axs = plt.subplots(3)
axs[0].plot(
    x.training_set.DATE, x.training_set.future, label="Realized Volatility", alpha=0.5
)
axs[0].plot(x.training_set.DATE, prediction, label="Prediction", alpha=0.8)
axs[0].legend()
axs[1].plot(x.training_set.future, prediction, "o", alpha=0.4)
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
)
