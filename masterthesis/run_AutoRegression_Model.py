from AutoRegression_Model import AutoRegressionModel
from run_HAR_model import *
from config import folder_structure


def load_data():
    df_m = pd.read_csv(
        folder_structure.path_input + "/" + "RealizedMeasures03_10.csv", index_col=0
    )
    df_m.DATE = df_m.DATE.values
    df_m.DATE = pd.to_datetime(df_m.DATE, format="%Y%m%d")

    return df_m


def load_validation_data():
    df = pd.read_csv(
        folder_structure.path_input + "/" + "DataFeatures.csv", index_col=0
    )
    df.DATE = df.DATE.values
    df.DATE = pd.to_datetime(df.DATE, format="%Y%m%d")

    return df


def estimate_and_predict_ar_models(df_input, save: bool = False):
    all_models = {"future": [1, 5, 20], "lag": [1, 3]}
    all_results = {}

    os.chdir(folder_structure.output_AR)

    for i in all_models["future"]:
        for k in all_models["lag"]:
            all_results["AR_{}_{}".format(i, k)] = AutoRegressionModel(
                df=df_input,
                future=i,
                ar_lag=k,
                period_train=list(
                    [
                        pd.to_datetime("20030910", format="%Y%m%d"),
                        pd.to_datetime("20080208", format="%Y%m%d"),
                    ]
                ),
                period_test=list(
                    [
                        pd.to_datetime("20080209", format="%Y%m%d"),
                        pd.to_datetime("20101231", format="%Y%m%d"),
                    ]
                ),
            )
            all_results["AR_{}_{}".format(i, k)].make_accuracy()

            if save:
                accuracy_results = open("AR_{}_{}_accuracy.txt".format(i, k), "a+")
                accuracy_results.write("Train Accuracy:")
                accuracy_results.write(
                    str(all_results["AR_{}_{}".format(i, k)].train_accuracy)
                )
                accuracy_results.write("Test Accuracy:")
                accuracy_results.write(
                    str(all_results["AR_{}_{}".format(i, k)].test_accuracy)
                )

    return all_results


def run_all(save_output: bool = False):
    df = load_data()
    res = estimate_and_predict_ar_models(df_input=df, save=save_output)

    return res


results_ar = run_all(save_output=False)
