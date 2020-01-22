from run_HAR_model import load_data
from LSTM import *
from collections import OrderedDict


"""
- Make Summary Statistics for the whole population (eg. conditional average and so on) --> call it, NETWORK ANALYSIS
- how many epochs to choose? (chose epochs such that RMSE is becoming stable in the testing set
- produce over-fitted model, to prove that the methodology actually has potential

"""


class GeneticAlgorithm:
    def __init__(
        self,
        training_set,
        testing_set,
        network_architecture=OrderedDict(
            [
                ("Layer1", [10, 21, 5]),
                ("Layer2", [2, 20, 5]),
                ("Layer3", [1, 21, 1]),
                ("Layer4", [0, 5, 1]),
            ]
        ),
        learning_rate=[0.0001, 0.1, 0.005],
        batch_size=[20, 50, 5],
        initial_poulation_source_external=False,  # if there is a csv file with the initial population (load it into the algorithm)
    ):
        self.training_set = training_set
        self.testing_set = testing_set
        self.network_architecture = network_architecture
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.initial_poulation_source_external = initial_poulation_source_external

        self.initial_population = None
        self.parent_one = None
        self.parent_two = None

    def make_initial_population(self, save_population_to_csv=False):

        if self.initial_poulation_source_external:
            pass  # define path were the csv file is stored // plus define a

        else:
            complete_architecture = OrderedDict(
                [
                    (
                        "LR",
                        np.arange(
                            self.learning_rate[0],
                            self.learning_rate[1],
                            self.learning_rate[2],
                        ),
                    ),
                    (
                        "BS",
                        np.arange(
                            self.batch_size[0], self.batch_size[1], self.batch_size[2]
                        ),
                    ),
                ]
            )

            for i in range(len(self.network_architecture.keys())):
                complete_architecture.update(
                    {
                        list(self.network_architecture.keys())[i]: np.arange(
                            self.network_architecture[
                                list(self.network_architecture.keys())[i]
                            ][0],
                            self.network_architecture[
                                list(self.network_architecture.keys())[i]
                            ][1],
                            self.network_architecture[
                                list(self.network_architecture.keys())[i]
                            ][2],
                        )
                    }
                )

            self.network_architecture = complete_architecture

            ind = pd.DataFrame(
                0,
                index=range(5),
                columns=np.array(
                    list(self.network_architecture.keys()) + ["Fitness", "Generation"]
                ),
            )

            for i in range(
                ind.shape[0]
            ):  # this loop is quite slow: might be better to make dictionary out of it
                for j in range(len(self.network_architecture.keys())):
                    ind[list(self.network_architecture.keys())[j]].iloc[
                        i
                    ] = random.choice(
                        self.network_architecture[
                            list(self.network_architecture.keys())[j]
                        ]
                    )

            self.initial_population = ind

            for i in range(self.initial_population.shape[0]):
                tf.reset_default_graph()
                training_model = TrainLSTM(
                    self.training_set,
                    self.testing_set,
                    activation=tf.nn.elu,
                    epochs=1,
                    learning_rate=self.initial_population.LR[i],
                    network_architecture=self.initial_population[
                        ["Layer1", "Layer2", "Layer3", "Layer4"]
                    ]
                    .iloc[i]
                    .to_dict(),
                )
                training_model.make_accuracy_measures()

                self.initial_population.Fitness[i] = training_model.fitness

            if save_population_to_csv:
                pass  # save initial_population to a given path (as csv file)

    def select_parents(
        self,
    ):  # add two different methods to select parents in __init__()
        if self.initial_population is None:
            self.make_initial_population()

        df_help = self.initial_population.copy()
        df_help.Fitness = df_help.Fitness  # ** 2
        prob_dist = df_help.Fitness / np.sum(df_help.Fitness)
        arr_individuals = np.arange(df_help.index[0], df_help.shape[0])
        parent1_location = np.random.choice(arr_individuals, p=prob_dist)

        ind_help = df_help.drop(parent1_location, axis=0)

        prob_dist = ind_help.Fitness / np.sum(ind_help.Fitness)
        arr_individuals = np.delete(arr_individuals, parent1_location)
        parent2_location = np.random.choice(arr_individuals, p=prob_dist)

        parent1 = df_help.iloc[parent1_location]
        parent2 = df_help.iloc[parent2_location]

        self.parent_one = parent1
        self.parent_two = parent2

    def make_offsprings(self):
        if self.parent_one is None:
            self.select_parents()

        df_test_parent_one = self.parent_one.to_frame()
        df_test_parent_two = self.parent_two.to_frame()
        df_test_parent_one = (
            df_test_parent_one.transpose().reset_index(drop=True).reset_index(level=0)
        )
        df_test_parent_two = (
            df_test_parent_two.transpose().reset_index(drop=True).reset_index(level=0)
        )

        child_one = df_test_parent_one[list(df_test_parent_one.columns[:3])].merge(
            df_test_parent_two[list(df_test_parent_one.columns[3:7])],
            on=df_test_parent_one.index,
        )
        child_two = df_test_parent_two[list(df_test_parent_one.columns[:3])].merge(
            df_test_parent_one[list(df_test_parent_one.columns[3:7])],
            on=df_test_parent_one.index,
        )

        child_one = child_one.drop(["key_0", "index"], axis=1)
        child_two = child_two.drop(["key_0", "index"], axis=1)

        _random_pick = random.choice(child_one.columns)
        child_three = child_one.copy()
        child_three[_random_pick] = random.choice(
            self.network_architecture[_random_pick]
        )

        _random_pick = random.choice(child_two.columns)  # here might be a mistake !!
        child_four = child_two.copy()
        child_four[_random_pick] = random.choice(
            self.network_architecture[_random_pick]
        )

        self.initial_population = self.initial_population.append(child_one, sort=False)
        self.initial_population = self.initial_population.append(child_two, sort=False)
        self.initial_population = self.initial_population.append(
            child_three, sort=False
        )
        self.initial_population = self.initial_population.append(child_four, sort=False)

        self.initial_population.reset_index(drop=True, inplace=True)

        ind = self.initial_population[
            self.initial_population.Fitness.isna()
        ].reset_index(drop=True)

        for i in ind.index:  # error !!
            tf.reset_default_graph()
            training_model = TrainLSTM(
                self.training_set,
                self.testing_set,
                activation=tf.nn.elu,
                epochs=1,
                learning_rate=ind.LR[i],
                network_architecture=ind[["Layer1", "Layer2", "Layer3", "Layer4"]]
                .iloc[i]
                .to_dict(),
            )
            training_model.make_accuracy_measures()

            self.initial_population.Fitness[i] = training_model.fitness


df = load_data()

_lstm = DataPreparationLSTM(df=df, future=1)
_lstm.prepare_complete_data_set()

_ga = GeneticAlgorithm(_lstm.training_set, _lstm.testing_set)
_ga.make_initial_population()

_ga.select_parents()
_ga.make_offsprings()
_ga.initial_population

check = _ga.initial_population
check
ind = check[check.Fitness.isna()]
ind

for i in ind.index:
    print(i)

u = TrainLSTM(
    _lstm.training_set,
    _lstm.testing_set,
    learning_rate=ind.LR[0],
    epochs=1,
    network_architecture={"Layer3": 5.0, "Layer2": 2.0, "Layer1": 15.0, "Layer4": 0.0},
)
u.train_lstm()

u.make_accuracy_measures()


ind[["Layer1", "Layer2", "Layer3", "Layer4", "Layer5"]].iloc[0].to_dict()
ind.LR[0]

for i in range(ind.shape[0]):
    tf.reset_default_graph()

    training_model_u = TrainLSTM(
        _ga.training_set,
        _ga.testing_set,
        activation=tf.nn.elu,
        epochs=1,
        learning_rate=ind.LR[i],
        network_architecture=ind[["Layer1", "Layer2", "Layer3", "Layer4", "Layer5"]]
        .iloc[i]
        .to_dict(),
    )

    training_model_u.make_accuracy_measures()
    ind.Fitness[i] = training_model_u.fitness


check = _ga.initial_population
check.head()
