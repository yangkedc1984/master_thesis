from run_HAR_model import *
from LSTM import *
from config import *
from HAR_Model import HARModel


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

plt.plot(
    results["har_20_True"].training_set.DATE, results["har_20_True"].training_set.future
)
plt.plot(
    results["har_20_True"].training_set.DATE,
    results["har_20_True"].prediction_train,
    color="black",
)
plt.plot(
    results["har_5_True"].training_set.DATE,
    results["har_5_True"].prediction_train,
    alpha=0.5,
)
plt.close()

_training_set = results["har_20_True"].training_set
_training_set = _training_set.iloc[0:30]


_training_set.RV_t.iloc[1:21].mean() - _training_set.future[0]  # test passed


def load_data():  # loading realized measures (not the entire data set)
    df_m = pd.read_csv(
        instance_path.path_input + "/" + "RealizedMeasures03_10.csv", index_col=0
    )
    df_m.DATE = df_m.DATE.values
    df_m.DATE = pd.to_datetime(df_m.DATE, format="%Y%m%d")

    return df_m


df = load_data()

import matplotlib.pyplot as plt
import numpy as np

df.RV = np.where(df.RV > 0.001, 0.001, df.RV)

plt.plot(df.DATE, df.RV)

plt.hist(df.RV, bins=100)
plt.close()


len(df.RV[df.RV > 0.001])  # 34 observations are extreme

import numpy as np
import pandas as pd


df_m = load_data()
df = df_m[["DATE", "RV"]].copy()
df["threshold"] = df["RV"].rolling(window=200).std() * 4  # threshold
df.threshold = np.where(df.threshold.isna(), 1, df.threshold)
df["larger"] = np.where(df.RV > df.threshold, True, False)
df = df[df.larger == False]

# count number of TRUE in sample
df.larger.value_counts()
df.shape

(51 / 1842) * 100  # 2.7% of the sample is deleted

df = df[df.larger == False]


plt.plot(df.DATE[300:500], df.RV[300:500], color="navy", alpha=0.5)
plt.plot(df.DATE[300:500], df.threshold[300:500], color="black", alpha=0.5)
plt.close()

x = HARModel(df, future=1)
x.run_complete_model()
x.make_graph()

plt.plot(
    x.training_set.DATE, x.prediction_train, color="black", label="Predicted Volatility"
)
plt.plot(
    x.training_set.DATE,
    x.training_set.future,
    color="green",
    alpha=0.5,
    label="Realized Volatility",
)
plt.close()
