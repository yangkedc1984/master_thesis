"""
Realized Volatility is computed here (only the basic version though)
    - Update: Check how outliers and extreme values are treated
    - Update: Check what re-sampling does and whether it improves the results (10 fold business time sampling)

"""

from config import *
import pandas as pd
import numpy as np
from tqdm import tqdm

tqdm.pandas()


def load_raw_data():
    hf_data = pd.read_csv(
        instance_path.path_input
        + "/"
        + "SPY2011.csv",  # nrows=10000000  # use nrows for only a selection of the data
    )

    return hf_data


def etl(df):
    df = df.drop(["SYM_ROOT", "SYM_SUFFIX"], axis=1)

    # creating sub_df which excludes trades that are out of opening hours
    df.TIME_M = pd.to_timedelta(df.TIME_M, "ns")
    cut_l = pd.to_timedelta("09:30:00")
    cut_u = pd.to_timedelta("16:00:00")
    df = df.loc[(df.TIME_M >= cut_l) & (df.TIME_M <= cut_u)]

    # sampling [--> sample 10 times and take averages (increases computational stress)]
    interval_size = np.floor(df.shape[0] / 78)
    interval = np.arange(0, df.shape[0], interval_size)
    df = df.iloc[interval]

    # Log Prices, Returns & Realized Volatility
    df["LogPrice"] = np.log(df["PRICE"])
    df["Returns"] = df["LogPrice"] - df["LogPrice"].shift(1)
    df["RV"] = df["Returns"] ** 2

    # realized semi variance squared positives
    df["RSV_plus"] = np.nan
    df.RSV_plus = (df.Returns.loc[df.Returns > 0]) ** 2
    df.loc[np.isnan(df.RSV_plus), "RSV_plus"] = 0

    # realized semi variance squared negatives
    df["RSV_minus"] = np.nan
    df.RSV_minus = (df.Returns.loc[df.Returns < 0]) ** 2
    df.loc[np.isnan(df.RSV_minus), "RSV_minus"] = 0

    rv = df["RV"].sum() * 100
    rsv_plus = (
        df["RSV_plus"].sum() * 100
    )  # is * 100 an issue? (not done in previous verison)
    rsv_minus = df["RSV_minus"].sum() * 100

    series_help = pd.Series([rv, rsv_plus, rsv_minus])

    return series_help  # df_output  # pd.Series([rv, rsv_plus, rsv_minus])


def make_all_features(high_frequency_data_set):
    df = high_frequency_data_set.groupby(high_frequency_data_set.DATE).progress_apply(
        lambda x: etl(x)
    )
    df.rename(columns={0: "RV", 1: "RSV_plus", 2: "RSV_minus"}, inplace=True)
    df.reset_index(level=0, inplace=True)

    return df


def save_data_features(df):
    df.to_csv(
        instance_path.path_input + "/" + "DataFeatures.csv"
    )  # adding a unique identifier?
    print("Data exported in {}".format(instance_path.path_input))


def run_feature_engineering():
    hf_data = load_raw_data()
    df = make_all_features(hf_data)
    save_data_features(df)

    # return df


run_feature_engineering()
