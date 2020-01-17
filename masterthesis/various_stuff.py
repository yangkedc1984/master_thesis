from run_HAR_model import *
from LSTM import *
from config import *


plt.style.use("seaborn")


results["har_1_True"].make_graph()

print(results["har_20_True"].estimation_results)


for i in [1, 5, 20]:
    plt.plot(
        results["har_{}_True".format(i)].training_set.DATE,
        results["har_{}_True".format(i)].prediction_train,
        label="{} day".format(i),
        alpha=0.5,
    )
plt.legend()
plt.close()

_training_set = results["har_20_True"].training_set
_training_set = _training_set.iloc[0:30]


_training_set.RV_t.iloc[1:21].mean() - _training_set.future[0]  # test passed


def load_data():
    df_m = pd.read_csv(
        instance_path.path_input + "/" + "RealizedMeasures03_10.csv", index_col=0
    )
    df_m.DATE = df_m.DATE.values
    df_m.DATE = pd.to_datetime(df_m.DATE, format="%Y%m%d")

    return df_m


df = load_data()

# this works perfectly fine
# lstm_instance = LSTM(df=df, future=20, semi_variance=False)
# lstm_instance.generate_complete_data_set()
#
# data_test = lstm_instance.df_processed_data
# data_test.head()


def sequence_future(_series, num_lag, num_future):
    # Input Variables
    num_1 = num_lag  # the number of lags
    num_2 = num_future  # the future average you want to compute
    names = ["Lag" + str((num_lag - i)) for i in range(num_1)]
    name_depend = "future_average_of_{}days".format(num_2)
    names.append(name_depend)  # names of the data frame

    # Input Data
    _rv = _series.RV
    k = pd.DataFrame(0, index=range(len(_rv) - num_1 - num_2), columns=range(num_1 + 1))

    # Generating Sequence
    for i in range(num_1, len(_rv) - num_2):
        k.iloc[i - num_1] = _rv[(i - num_1) : i].reset_index(drop=True)
        k.iloc[(i - num_1)][num_lag] = np.mean(
            _rv[i : (i + num_2)].reset_index(drop=True)
        )

    k.columns = names

    # Returning Data Frame
    return k


import numpy as np

df_old_function = sequence_future(df, 20, 20)
df_seq = df_old_function.rename(
    index=str, columns={df_old_function.columns[df_old_function.shape[1] - 1]: "y"}
)
x_train = df_seq.drop(["y"], axis=1)
x_train.values
