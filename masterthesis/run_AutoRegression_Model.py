from AutoRegression_Model import *
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
                df=df_input, future=i, ar_lag=k,
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


def validation_accuracy(
    df_input, model_dictionaries, save: bool = False, return_results: bool = False
):
    os.chdir(folder_structure.output_AR)

    all_models = {"future": [1, 5, 20], "lag": [1, 3]}

    for i in all_models["future"]:
        for k in all_models["lag"]:
            validation_instance = TimeSeriesDataPreparationLSTM(
                df=df_input,
                future=i,
                lag=k,
                standard_scaler=False,
                min_max_scaler=False,
                log_transform=False,
                semi_variance=False,
                jump_detect=True,
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
            validation_instance.prepare_complete_data_set()
            validation_set = validation_instance.training_set

            if k == 1:
                prediction = model_dictionaries[
                    "AR_{}_{}".format(i, k)
                ].ar_model.predict(validation_set["RV"])

            else:
                prediction = model_dictionaries[
                    "AR_{}_{}".format(i, k)
                ].ar_model.predict(validation_set[["RV", "lag_1", "lag_2"]])

            train_accuracy = {
                "MSE": metrics.mean_squared_error(validation_set["future"], prediction),
                "MAE": metrics.mean_absolute_error(
                    validation_set["future"], prediction
                ),
                "RSquared": metrics.r2_score(validation_set["future"], prediction),
            }

            if save:
                accuracy_results = open("AR_{}_{}_accuracy.txt".format(i, k), "a+")
                accuracy_results.write("Validation Accuracy:")
                accuracy_results.write(str(train_accuracy))

            if return_results:
                return train_accuracy


def run_all(save_output: bool = False):
    df = load_data()
    df_validation = load_validation_data()
    res = estimate_and_predict_ar_models(df_input=df, save=save_output)
    validation_accuracy(
        df_input=df_validation,
        model_dictionaries=res,
        save=save_output,
        return_results=False,
    )

    return res


results_auto_regression = run_all(save_output=True)
