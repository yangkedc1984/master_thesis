from GeneticAlgorithm import *
from config import *
from run_HAR_model import load_data

# load data
df = load_data()

# prepare LSTM data
_lstm_instance = DataPreparationLSTM(
    df=df,
    future=20,
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
_ga_1 = GeneticAlgorithm(
    training_set_ga=_lstm_instance.training_set,
    testing_set_ga=_lstm_instance.testing_set,
    network_architecture=OrderedDict(
        [
            ("Layer1", [2, 43, 5]),
            ("Layer2", [1, 42, 4]),
            ("Layer3", [0, 21, 3]),
            ("Layer4", [0, 21, 4]),
        ]
    ),
    learning_rate=[0.00001, 0.2, 0.005],
    initial_population_source_external=False,
    build_grid_scenarios=True,
)
_ga_1.run_complete_genetic_algorithm(number_of_generations=50)


result = _ga_1.initial_population.copy()
result.to_csv(
    folder_structure.path_input + "/" + "InitialPopulation_sv_20_newfitness.csv"
)


# df_1 = pd.read_csv(
#     instance_path.path_input + "/" + "InitialPopulation_sv_1_3.csv", index_col=0
# )
# df_5 = pd.read_csv(
#    instance_path.path_input + "/" + "InitialPopulation_sv_5.csv", index_col=0
# )
# df_20 = pd.read_csv(
#     instance_path.path_input + "/" + "InitialPopulation_sv_20.csv", index_col=0
# )
#
# df_run = pd.read_csv(
#     instance_path.path_input + "/" + "InitialPopulation_run_test.csv", index_col=0
# )
