# Playground
from run_LSTM import *
from HAR_Model import HARModel

df = load_data()

har_data_prep = HARModel(
    df=df,
    future=1,
    semi_variance=True,
    jump_detect=True,
    scaling=True,
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
har_data_prep.make_testing_training_set()
check = har_data_prep.training_set

output_test = TrainLSTM(
    training_set=har_data_prep.training_set,
    testing_set=har_data_prep.testing_set,
    epochs=20,
    learning_rate=0.001,
    layer_one=40,
    layer_two=20,
    layer_three=10,
    layer_four=2,
)
output_test.make_accuracy_measures()

output_test.make_performance_plot(show_testing_sample=True)

train_real = np.exp(output_test.training_set.future)
train_pred = np.exp(output_test.prediction_train)

test_real = np.exp(output_test.testing_set.future)
test_pred = np.exp(output_test.prediction_test)

metrics.mean_squared_error(train_real, train_pred)
metrics.mean_absolute_error(train_real, train_pred)
metrics.r2_score(train_real, train_pred)

metrics.mean_squared_error(test_real, test_pred)
metrics.mean_absolute_error(test_real, test_pred)
metrics.r2_score(test_real, test_pred)


plt.close()
fig, axs = plt.subplots(2)
axs[0].plot(
    output_test.training_set.DATE, train_real, lw=0.5, label="Realized Volatility"
)
axs[0].plot(output_test.training_set.DATE, train_pred, lw=0.5, label="Prediction")
axs[0].legend()

axs[1].plot(
    output_test.testing_set.DATE, test_real, lw=0.5, label="Realized Volatility"
)
axs[1].plot(output_test.testing_set.DATE, test_pred, lw=0.5, label="Prediction")
axs[1].legend()


### VALIDATION SET

df_v = pd.read_csv(folder_structure.path_input + "/" + "DataFeatures.csv", index_col=0)
df_v.DATE = df.DATE.values
df_v.DATE = pd.to_datetime(df.DATE, format="%Y%m%d")

har_validation = HARModel(
    df=df_v,
    future=1,
    jump_detect=True,
    scaling=True,
    semi_variance=True,
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
har_validation.make_testing_training_set()


train_matrix = har_validation.training_set.drop(columns={"DATE", "future"}).values
train_shape_rows = train_matrix.shape[0]
train_shape_columns = train_matrix.shape[1]

train_matrix = train_matrix.reshape((train_shape_rows, train_shape_columns, 1))

validation_pred = np.exp(output_test.fitted_model.predict(train_matrix))
validation_real = np.exp(har_validation.training_set.future)

plt.close()
plt.plot(har_validation.training_set.DATE, validation_real, lw=0.5, label="real")
plt.plot(har_validation.training_set.DATE, validation_pred, lw=0.5, label="pred")
plt.legend()


metrics.r2_score(validation_real, validation_pred)
metrics.mean_absolute_error(validation_real, validation_pred)
metrics.mean_squared_error(validation_real, validation_pred)


"""
class TrainLSTM:
    def __init__(
        self,
        training_set,
        testing_set,
        activation=tf.nn.elu,
        epochs=20,
        learning_rate=0.001,
        layer_one=20,
        layer_two=15,
        layer_three=8,
        layer_four=4,
    ):
"""

x = AutoRegressionModel(df=df, future=5, ar_lag=3)
x.estimate_model()
x.make_accuracy()
x.train_accuracy

# Fitness Function
def fitness_final(output, validation_data):
    """
    Fitness function that counts the number of observations in a given interval around the realized volatility
    :param output: array of prediction (which shape?)
    :param validation_data: array of validation data (which shape?)
    :return: fitness value function that maps prediction and actuals to a number of (0, 1)
    :return: lower and upper bound vectors
    """
    _margin_0 = (
        np.std(validation_data) / 10
    )  # can you do this on a rolling window of 200 days?
    _margin_1 = 0.05

    _gamma_0 = 0 - _margin_0
    _delta_0 = 0 + _margin_0

    _gamma_1 = 1 - _margin_1
    _delta_1 = 1 + _margin_1

    _lower_vector = _gamma_0 + _gamma_1 * validation_data
    _upper_vector = _delta_0 + _delta_1 * validation_data

    count_total = np.where((_lower_vector < output) & (_upper_vector > output), 1, 0,)
    np.sum(count_total) / len(prediction)

    fitness = np.sum(count_total) / len(output)

    return fitness, _lower_vector, _upper_vector


actuals = lstm_instance.back_transformation(
    np.array(x.training_set.future).reshape(-1, 1)
)
prediction = lstm_instance.back_transformation(
    np.array(x.prediction_train).reshape(-1, 1)
)

fitness_final(prediction, actuals)[0]

plt.close()
plt.plot(lstm_instance.training_set.DATE, prediction)
plt.plot(lstm_instance.training_set.DATE, actuals)
plt.plot(lstm_instance.training_set.DATE, fitness_final(prediction, actuals)[1], lw=0.5)
plt.plot(lstm_instance.training_set.DATE, fitness_final(prediction, actuals)[2], lw=0.5)
