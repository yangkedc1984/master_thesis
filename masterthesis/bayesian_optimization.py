print("CHECK 50")

from run_HAR_model import load_data
from LSTM import *
from functools import partial
from bayes_opt import BayesianOptimization
from config import *


df_input = load_data()

lstm_instance = TimeSeriesDataPreparationLSTM(
    df=df_input,
    future=1,
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


def fit_with_for_bayesian(lr_for_optimization, layer_1, layer_2, layer_3, layer_4):
    model = TrainLSTM(
        training_set=lstm_instance.training_set,
        testing_set=lstm_instance.testing_set,
        epochs=50,
        learning_rate=lr_for_optimization,
        layer_one=max(int(layer_1 * 15), 2),
        layer_two=max(int(layer_2 * 15), 2),
        layer_three=max(int(layer_3 * 15), 0),
        layer_four=max(int(layer_4 * 15), 0),
        adam_optimizer=True,
    )
    model.make_accuracy_measures()

    return model.fitness


fit_with_partial = partial(fit_with_for_bayesian)


# Bounded region of parameter space
pbounds = {
    "lr_for_optimization": (1e-3, 1e-2),
    "layer_1": (0.1, 3.1),
    "layer_2": (0.1, 3.1),
    "layer_3": (0, 1.5),
    "layer_4": (0, 1.5),
}

optimizer = BayesianOptimization(
    f=fit_with_partial, pbounds=pbounds, verbose=2, random_state=1,
)

init_points = 20
n_iter = 50
optimizer.maximize(
    init_points=init_points, n_iter=n_iter,
)

target_list = list([])
for i in range(init_points + n_iter):
    target_list.append(optimizer.res[i]["target"])


results_iterative = pd.Series(target_list, name="value")
results_iterative.to_csv(
    folder_structure.output_Bayesian
    + "/"
    + "bayesian_results_iterative_{}_{}_{}.csv".format("RV", 1, 20)
)


results_max = pd.DataFrame(optimizer.max)
results_max.to_csv(
    folder_structure.output_Bayesian
    + "/"
    + "bayesian_results{}_{}_{}.csv".format("RV", 1, 20)
)
