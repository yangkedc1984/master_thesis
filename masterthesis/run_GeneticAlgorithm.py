from GeneticAlgorithm import *
from config import *

# load data
df = load_data()

# prepare LSTM data
_lstm_instance = DataPreparationLSTM(
    df=df,
    future=1,
    lag=20,
    standard_scaler=False,
    min_max_scaler=True,
    log_transform=True,
    semi_variance=True,
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
_lstm_instance.prepare_complete_data_set()

# Run Genetic Algorithm
_ga = GeneticAlgorithm(
    training_set_ga=_lstm_instance.training_set,
    testing_set_ga=_lstm_instance.testing_set,
    network_architecture=OrderedDict(
        [
            ("Layer1", [6, 41, 2]),
            ("Layer2", [2, 21, 2]),
            ("Layer3", [0, 21, 5]),
            ("Layer4", [0, 21, 5]),
        ]
    ),
    learning_rate=[0.00001, 0.1, 0.005],
    batch_size=[20, 50, 5],
    initial_population_source_external=True,
)


_ga.run_complete_genetic_algorithm(initial_population_size=20, number_of_generations=20)


result = _ga.initial_population.copy()
result.to_csv(instance_path.path_input + "/" + "InitialPopulation_run_test.csv")

print("RESULTS SAVED")
