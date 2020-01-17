from run_HAR_model import *
from LSTM import *

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


lstm_instance = LSTM(df=df, future=20)
lstm_instance.generate_complete_data_set()
data_test = lstm_instance.df_processed_data

df_to = df[["DATE", "RSV_minus"]]
df_test = data_test.merge(df_to, on="DATE")
