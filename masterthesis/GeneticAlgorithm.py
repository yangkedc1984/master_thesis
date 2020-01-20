from collections import OrderedDict
from run_HAR_model import load_data
from LSTM import *
from run_LSTM import *


class GeneticAlgorithm:
    def __init__(
        self,
        data_frame,
        network_architecture=OrderedDict(
            [
                ("Layer1", [10, 20, 5]),
                ("Layer2", [2, 20, 5]),
                ("Layer3", [1, 20, 1]),
                ("Layer4", [0, 5, 1]),
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
                    list(self.network_architecture.keys())[
                        i
                    ]: self.network_architecture[
                        list(self.network_architecture.keys())[i]
                    ]
                }
            )

        self.network_architecture = complete_architecture

        ind = pd.DataFrame(
            0,
            index=range(40),
            columns=np.array(
                list(x.network_architecture.keys()) + ["Fitness", "Generation"]
            ),
        )

        for i in range(ind.shape[0]):

            for j in range(len(self.network_architecture.keys())):
                ind[list(self.network_architecture.keys())[j]][i] = float(
                    random.choice(
                        self.network_architecture[
                            list(self.network_architecture.keys())[j]
                        ]
                    )
                )

        self.initial_population = ind

        TrainLSTM()


df = load_data()

x = GeneticAlgorithm(data_frame=df)
x.make_initial_population()
x.initial_population.LR


x.network_architecture["LR"]

random.choice(x.network_architecture["LR"])
x.initial_population.LR = random.choice(x.network_architecture["LR"])
x.initial_population


# def initial_population_lstm(
#     size, x_train_input, y_train_input, x_test_input, y_test_input
# ):
#     pool = np.array(
#         [
#             np.arange(0.00001, 0.001, 0.00005),  # learning rate
#             np.arange(5, 51, 5),  # hidden layer 1
#             np.arange(3, 33, 2),  # hidden layer 2
#             np.arange(0, 30, 1),  # hidden layer 3
#             np.arange(0, 30, 1),  # hidden layer 3
#             np.arange(20, 50, 5),  # batch size
#         ]
#     )
#
#     ind = pd.DataFrame(
#         0,
#         index=range(size),
#         columns=np.array(["lr", "N1", "N2", "N3", "N4", "BS", "F", "Gen"]),
#     )
#     _output = list([])
#
#     for i in range(ind.shape[0]):
#         # time.sleep(2.0)
#
#         ind.Gen.iloc[i] = 0
#         ind.lr.iloc[i] = random.choice(pool[0])
#         ind.N1.iloc[i] = random.choice(pool[1])
#         ind.N2.iloc[i] = random.choice(pool[2])
#         ind.N3.iloc[i] = random.choice(pool[3])
#         ind.N4.iloc[i] = random.choice(pool[4])
#         ind.BS.iloc[i] = random.choice(pool[5])
#
#         out_nn = lstm_neural_network(
#             x_train_input,
#             y_train_input,
#             x_test_input,
#             y_test_input,
#             ind.lr.iloc[i],
#             ind.N1.iloc[i],
#             ind.N2.iloc[i],
#             ind.N3.iloc[i],
#             ind.N4.iloc[i],
#             ind.BS.iloc[i],
#             10,
#         )
#         ind.F.iloc[i] = out_nn[0] * out_nn[1]
#
#         _output.append(
#             np.array(
#                 [
#                     out_nn[8].predict(x_test_input),
#                     0,
#                     out_nn[0],  # x_test_s is the scaled testing sample
#                     out_nn[1],
#                     out_nn[0] * out_nn[1],
#                 ]
#             )
#         )
#
#         print(
#             "{} percent of the data set has been generated.".format(
#                 (i + 1) / ind.shape[0] * 100
#             )
#         )
#
#     return ind, pool, _output
