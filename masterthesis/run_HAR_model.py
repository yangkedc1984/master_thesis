from config import *
from HAR_Model import *


def load_data():
    df_m = pd.read_csv(
        folder_structure.path_input + "/" + "RealizedMeasures03_10.csv", index_col=0
    )
    df_m.DATE = df_m.DATE.values
    df_m.DATE = pd.to_datetime(df_m.DATE, format="%Y%m%d")

    return df_m


def estimate_and_predict_har_models(df_input, save=True):
    all_models = {"future": [1, 5, 20], "semi_variance": [True, False]}
    all_results = {}

    os.chdir(folder_structure.output_path + "/" + folder_structure.HARModel)

    for i in all_models["future"]:
        for k in all_models["semi_variance"]:
            for log_t in [True, False]:
                all_results[
                    "har_{}_{}_{}".format(i, k, log_t)
                ] = HARModelLogTransformed(
                    df=df_input,
                    future=i,
                    lags=[4, 20],
                    feature="RV",
                    semi_variance=k,
                    jump_detect=True,
                    log_transformation=log_t,
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
                all_results["har_{}_{}_{}".format(i, k, log_t)].run_complete_model()

                if save:
                    estimation_results = open(
                        "har_{}_{}_{}_estimation.txt".format(i, k, log_t), "a+"
                    )
                    estimation_results.write(
                        all_results[
                            "har_{}_{}_{}".format(i, k, log_t)
                        ].estimation_results
                    )

                    accuracy_results = open(
                        "har_{}_{}_{}_accuracy.txt".format(i, k, log_t), "a+"
                    )
                    accuracy_results.write("Train Accuracy:")
                    accuracy_results.write(
                        str(
                            all_results[
                                "har_{}_{}_{}".format(i, k, log_t)
                            ].train_accuracy
                        )
                    )
                    accuracy_results.write("Test Accuracy:")
                    accuracy_results.write(
                        str(
                            all_results[
                                "har_{}_{}_{}".format(i, k, log_t)
                            ].test_accuracy
                        )
                    )

    return all_results


def run_all(save_output=True):
    df = load_data()
    res = estimate_and_predict_har_models(df_input=df, save=save_output)

    return res


results_har = run_all(save_output=False)
