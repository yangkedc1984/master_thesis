from run_HAR_model import *
from config import *
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn")


df_m = pd.read_csv(
    instance_path.path_input + "/" + "RealizedMeasures03_10.csv", index_col=0
)
df_m.DATE = df_m.DATE.values
df_m.DATE = pd.to_datetime(df_m.DATE, format="%Y%m%d")

df_m.columns


df_m.RV_s.median()

plt.hist(df_m["RV_abs"], bins=500)
plt.hist(df_m["RV_s"], bins=500)

plt.plot(df_m.DATE, df_m.RV_abs)


results["har_5_True"].make_graph()

plt.plot(
    results["har_1_True"].prediction_test,
    results["har_1_True"].testing_set.future,
    "ko",
    alpha=0.5,
)


plt.hist(
    results["har_1_True"].prediction_test - results["har_1_True"].testing_set.future,
    bins=50,
    color="black",
    alpha=0.5,
)

hf_data = pd.read_csv(instance_path.path_input + "/" + "SPY2011.csv", nrows=1000000)

# hf_data.head(20)

import numpy as np


def clean_data_volume_based(input_df, indicator):
    input_df = input_df.drop(["SYM_ROOT", "SYM_SUFFIX"], axis=1)  # cleaning data frame
    sub_df = input_df.loc[input_df.DATE == input_df.DATE.unique()[indicator]]

    # creating sub_df which excludes trades that are out of opening hours
    sub_df.TIME_M = pd.to_timedelta(sub_df.TIME_M, "ns")
    cut_l = pd.to_timedelta("09:30:00")
    cut_u = pd.to_timedelta("16:00:00")

    sub_df = sub_df.loc[sub_df.TIME_M >= cut_l]  # add on one line of code!
    sub_df = sub_df.loc[sub_df.TIME_M <= cut_u]

    interval_size = np.floor(
        sub_df.shape[0] / 78
    )  # how can I choose the interval such that it changes over time
    interval = np.arange(0, sub_df.shape[0], interval_size)  # sample interval

    # compute measures
    df_m = sub_df.iloc[
        interval
    ]  # subset is selected from the whole data set (use this function to prep the data)

    return df_m


df_selected_interval = clean_data_volume_based(hf_data, 0)

hf_data.DATE.unique()


df_selected_interval.shape


def feature_gen(df_m):
    df_m["LogPrice"] = np.log(df_m.PRICE)
    df_m["Returns"] = df_m.LogPrice - df_m.LogPrice.shift(1)
    df_m = df_m.drop(df_m.index[0], axis=0)

    # realized volatility absolute and squared
    df_m["RV_abs"] = abs(df_m.Returns)
    df_m["RV_s"] = df_m.Returns ** 2

    # # realized semi variance absolute positives
    # df_m["RSV_a_plus"] = np.nan
    # df_m.RSV_a_plus = abs(df_m.Returns.loc[df_m.Returns > 0])
    # df_m.loc[np.isnan(df_m.RSV_a_plus), "RSV_a_plus"] = 0
    #
    # # realized semi variance absolute negatives
    # df_m["RSV_a_minus"] = np.nan
    # df_m.RSV_a_minus = abs(df_m.Returns.loc[df_m.Returns < 0])
    # df_m.loc[np.isnan(df_m.RSV_a_minus), "RSV_a_minus"] = 0
    #
    # # realized semi variance squared positives
    # df_m["RSV_s_plus"] = np.nan
    # df_m.RSV_s_plus = (df_m.Returns.loc[df_m.Returns > 0]) ** 2
    # df_m.loc[np.isnan(df_m.RSV_s_plus), "RSV_s_plus"] = 0
    #
    # # realized semi variance squared negatives
    # df_m["RSV_s_minus"] = np.nan
    # df_m.RSV_s_minus = (df_m.Returns.loc[df_m.Returns < 0]) ** 2
    # df_m.loc[np.isnan(df_m.RSV_s_minus), "RSV_s_minus"] = 0
    #
    df_m = df_m.drop(["TIME_M", "PRICE", "LogPrice", "Returns"], axis=1)

    return df_m


df_feature = feature_gen(df_selected_interval)


df_feature.RV_s.sum()

df_selected_interval["LogPrice"] = np.log(df_selected_interval.PRICE) * 100
df_selected_interval["ShiftedLogPrice"] = df_selected_interval.LogPrice.shift(1)
df_selected_interval["Returns"] = (
    df_selected_interval.LogPrice - df_selected_interval.ShiftedLogPrice
)

# df_selected_interval['Returns'] = df_selected_interval['Returns']
df_selected_interval["RV"] = df_selected_interval.Returns ** 2

df_selected_interval.RV.sum()

np.sqrt(df_selected_interval.RV.sum())

df_selected_interval.RV.sum()

df_feature.RV_s.sum() - df_selected_interval.RV.sum()

np.sqrt(df_selected_interval.RV.sum())

# make pandas transformations: groupby(DATE.unique()).lambda(df_m.RV.mean().sum()) ---> pseudo code
