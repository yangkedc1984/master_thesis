from collections import OrderedDict
from run_HAR_model import load_data
from LSTM import *


"""
additional optimization factor should be activation function
"""


class GeneticAlgorithm:
    def __init__(
        self,
        data_frame,
        network_architecture=OrderedDict(
            [
                ("Layer1", [10, 21, 5]),
                ("Layer2", [2, 20, 5]),
                ("Layer3", [1, 21, 1]),
                ("Layer4", [0, 5, 1]),
                ("Layer5", [0, 5, 1]),
            ]
        ),
        learning_rate=[0.0001, 0.01, 0.005],
        batch_size=[20, 50, 5],
    ):
        self.data_frame = data_frame
        self.network_architecture = network_architecture
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.initial_population = None

    def make_initial_population(self):
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
            index=range(40),
            columns=np.array(
                list(self.network_architecture.keys()) + ["Fitness", "Generation"]
            ),
        )

        for i in range(
            ind.shape[0]
        ):  # this loop is quite slow: might be better to make dictionary out of it
            for j in range(len(self.network_architecture.keys())):
                ind[list(self.network_architecture.keys())[j]].iloc[i] = random.choice(
                    self.network_architecture[list(self.network_architecture.keys())[j]]
                )

        self.initial_population = ind

        training_model = TrainLSTM()


df = load_data()

x = GeneticAlgorithm(data_frame=df)
x.make_initial_population()
check = x.initial_population


check[["Layer1", "Layer2", "Layer3", "Layer4", "Layer5"]].iloc[0].to_dict()
