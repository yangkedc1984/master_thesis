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
    epochs=10,
    learning_rate=0.5,
    layer_one=20,
    layer_two=10,
    layer_three=1,
    layer_four=1,
)
x.make_accuracy_measures()
x.fitness

x.make_performance_plot(show_testing_sample=False)

# x.fitted_model.save("LSTM_SV_1.h5")
# x.fitted_model.save("LSTM_RV_1.h5")
# x.fitted_model.save("LSTM_SV_5.h5")
# x.fitted_model.save("LSTM_RV_5.h5")
# x.fitted_model.save("LSTM_SV_20.h5")
# x.fitted_model.save("LSTM_RV_20.h5")

model_ = tf.keras.models.load_model("LSTM_SV_1.h5")
predict_model = model_.predict(lstm_instance.test_matrix)
predict_model = lstm_instance.back_transformation(predict_model)


metrics.mean_absolute_error(
    lstm_instance.back_transformation(
        np.array(lstm_instance.testing_set.future).reshape(-1, 1)
    ),
    predict_model,
) < metrics.mean_absolute_error(
    results["har_1_True"].testing_set.future, results["har_1_True"].prediction_test,
)

metrics.mean_squared_error(
    lstm_instance.back_transformation(
        np.array(lstm_instance.testing_set.future).reshape(-1, 1)
    ),
    predict_model,
) < metrics.mean_squared_error(
    results["har_1_True"].testing_set.future, results["har_1_True"].prediction_test,
)

from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(
    lstm_instance.back_transformation(x.prediction_train),
    lstm_instance.back_transformation(np.array(x.training_set.future).reshape(-1, 1)),
)
reg.coef_
reg.intercept_
reg.score(
    lstm_instance.back_transformation(x.prediction_train),
    lstm_instance.back_transformation(np.array(x.training_set.future).reshape(-1, 1)),
)

reg_har = LinearRegression().fit(
    np.array(results["har_20_True"].prediction_train).reshape(-1, 1),
    results["har_20_True"].training_set.future,
)
reg_har.coef_
reg_har.intercept_
reg_har.score(
    np.array(results["har_1_True"].prediction_train).reshape(-1, 1),
    results["har_1_True"].training_set.future,
)


plt.close()
plt.plot(
    np.array(results["har_20_True"].prediction_test).reshape(-1, 1),
    results["har_20_True"].testing_set.future,
    "o",
    alpha=0.3,
    color="darkred",
)
plt.plot(
    np.array(results["har_20_True"].prediction_train).reshape(-1, 1),
    results["har_20_True"].training_set.future,
    "o",
    alpha=0.1,
    color="black",
)


# performance plot that shows actual performance and then the bias of the two models --> indicates where LSTM works
# better than HAR and vice versa

plt.close()
fig, axs = plt.subplots(2)
axs[0].plot(
    lstm_instance.training_set.DATE,
    lstm_instance.back_transformation(x.prediction_train),
    label="Prediction LSTM",
    lw=1,
)
axs[0].plot(
    results["har_1_True"].training_set.DATE,
    results["har_1_True"].prediction_train,
    label="Prediction HAR",
    lw=1,
)
axs[0].plot(
    results["har_1_True"].training_set.DATE,
    results["har_1_True"].training_set.future,
    label="Realized Volatility",
    lw=0.5,
    color="black",
)
axs[0].legend()
axs[1].plot(
    results["har_1_True"].training_set.DATE,
    (
        results["har_1_True"].training_set.future
        - results["har_1_True"].prediction_train
    ),
    label="Error HAR",
    lw=0.5,
)
# axs[1].plot(
#     lstm_instance.training_set.DATE,
#     (
#         lstm_instance.back_transformation(
#             np.array(x.training_set.future).reshape(-1, 1)
#         )
#         - lstm_instance.back_transformation(x.prediction_train).reshape(
#             x.prediction_train.shape[0],
#         )
#     ),
#     label="Error LSTM",
#     lw=0.5,
# )
axs[1].legend()
