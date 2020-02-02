from run_HAR_model import *
from LSTM import *
from sklearn.decomposition import PCA
from config import *


df_input = load_data()

lstm_instance = DataPreparationLSTM(
    df=df_input,
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
lstm_instance.prepare_complete_data_set()

x = TrainLSTM(
    lstm_instance.training_set,
    lstm_instance.testing_set,
    epochs=10,
    learning_rate=0.05,
    layer_one=5,
    layer_two=2,
    layer_three=1,
    layer_four=1,
)
x.make_accuracy_measures()
x.fitness

x.make_performance_plot(show_testing_sample=True)
x.make_performance_plot(show_testing_sample=False)


# VALIDATION SET (NEVER SEEN OR USED FOR HYPERPARAMETER SETTING)
df_validation = pd.read_csv(
    instance_path.path_input + "/" + "DataFeatures.csv", index_col=0
)
df_validation.DATE = df_validation.DATE.values
df_validation.DATE = pd.to_datetime(df_validation.DATE, format="%Y%m%d")
df_validation.DATE.min()

lstm_validation = DataPreparationLSTM(
    df=df_validation,
    future=1,
    lag=20,
    standard_scaler=False,
    min_max_scaler=True,
    log_transform=True,
    semi_variance=True,
    jump_detect=True,
    period_train=list(
        [
            pd.to_datetime("20110101", format="%Y%m%d"),
            pd.to_datetime("20111231", format="%Y%m%d"),
        ]
    ),
    period_test=list(
        [
            pd.to_datetime("20110101", format="%Y%m%d"),
            pd.to_datetime("20111231", format="%Y%m%d"),
        ]
    ),
)
lstm_validation.prepare_complete_data_set()


train_matrix = lstm_validation.training_set.drop(columns={"DATE", "future"}).values
train_y = lstm_validation.training_set[["future"]].values

n_features = 1

train_shape_rows = train_matrix.shape[0]
train_shape_columns = train_matrix.shape[1]

train_matrix = train_matrix.reshape((train_shape_rows, train_shape_columns, n_features))

x.fitted_model.predict(train_matrix)

plt.plot(x.fitted_model.predict(train_matrix))
plt.plot(train_y)

plt.close()
plt.plot(x.fitted_model.predict(train_matrix), train_y, "o", color="black", alpha=0.25)


# check whether new parents are selected each time, or whether they stay the same... (would be a mistake)
# how to code that nicely


initial_population_scenarios = pd.read_csv(
    instance_path.path_input
    + "/"
    + "InitialPopulation_all_scenarios_future_1_newfitness.csv",
    index_col=0,
)
initial_population_scenarios = initial_population_scenarios.reset_index(level=0)
initial_population_scenarios.iloc[
    initial_population_scenarios.Fitness.nlargest(10).index
]

df_1 = pd.read_csv(
    instance_path.path_input + "/" + "InitialPopulation_sv_5_1.csv", index_col=0
)
df_2 = df_1.iloc[df_1.Fitness.nlargest(10).index]
df_2

# initial_population_scenarios = df_1.iloc[0:189]
# initial_population_scenarios = initial_population_scenarios.reset_index(level=0)

plt.close()
fig, axs = plt.subplots(2, 2)
axs[0, 0].hist(df_1.LR, bins=20)
axs[0, 1].hist(df_1.Layer1, bins=20)
axs[1, 0].hist(df_1.Layer2, bins=20)
axs[1, 1].hist(df_1.Layer3, bins=20)

df_1.iloc[df_1.Fitness.nlargest(20).index]
df_1.Fitness.nlargest(20).index

# make some conditional statement:
aver_learning = df_1[(df_1.Layer4 == 0) & (df_1.Layer3 == 0)]


plt.close()
plt.scatter(
    aver_learning.LR,
    aver_learning.Fitness,
    s=(np.mean(aver_learning[["Layer1", "Layer2"]], axis=1)) ** 2,
    marker="^",
    alpha=0.5,
)


# print PCA
pca = PCA(n_components=3)
pca.fit(initial_population_scenarios[["LR", "Layer1", "Layer2", "Layer3", "Layer4"]])
x = pd.DataFrame(
    pca.transform(
        initial_population_scenarios[["LR", "Layer1", "Layer2", "Layer3", "Layer4"]]
    ),
    columns={"one", "two", "three"},
).reset_index(level=0)
x = x.merge(initial_population_scenarios[["Fitness", "index"]], on="index")

pca_2 = PCA(n_components=3)
pca_2.fit(df_1[["LR", "Layer1", "Layer2", "Layer3", "Layer4"]])

x_2 = pd.DataFrame(
    pca.transform(df_1[["LR", "Layer1", "Layer2", "Layer3", "Layer4"]]),
    columns={"one", "two", "three"},
).reset_index(level=0)

x_2 = x_2.merge(
    df_1.reset_index(drop=True).reset_index(level=0)[["Fitness", "index"]], on="index"
)

pca_3 = PCA(n_components=3)
pca_3.fit(df_1[["LR", "Layer1", "Layer2", "Layer3", "Layer4"]])

x_3 = pd.DataFrame(
    pca.transform(df_2[["LR", "Layer1", "Layer2", "Layer3", "Layer4"]]),
    columns={"one", "two", "three"},
).reset_index(level=0)

x_3 = x_3.merge(
    df_2.reset_index(drop=True).reset_index(level=0)[["Fitness", "index"]], on="index"
)

from mpl_toolkits.mplot3d import Axes3D

plt.close()
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    x.one,
    x.two,
    x.three,
    c=x.Fitness,
    cmap="Greens",
    marker="^",
    s=200,  # c=x.Fitness, cmap="viridis",
)
ax.scatter(
    x_2.one,
    x_2.two,
    x_2.three,
    c=x_2.Fitness,
    cmap="binary",
    alpha=1,
    # s=(x_2.Fitness ** 4) / 1000000,
)
ax.scatter(
    x_3.one, x_3.two, x_3.three, c="red", alpha=0.2, s=(x_3.Fitness ** 4) / 5000000,
)
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_zlabel("PC 3")


# analysis with two components
plt.close()
pca = PCA(n_components=2)
pca.fit(initial_population_scenarios[["LR", "Layer1", "Layer2", "Layer3", "Layer4"]])
x = pd.DataFrame(
    pca.transform(
        initial_population_scenarios[["LR", "Layer1", "Layer2", "Layer3", "Layer4"]]
    ),
    columns={"one", "two"},
).reset_index(level=0)
x = x.merge(initial_population_scenarios[["Fitness", "index"]], on="index")
pca_2 = PCA(n_components=2)
pca_2.fit(df_1[["LR", "Layer1", "Layer2", "Layer3", "Layer4"]])
x_2 = pd.DataFrame(
    pca.transform(df_1[["LR", "Layer1", "Layer2", "Layer3", "Layer4"]]),
    columns={"one", "two"},
).reset_index(level=0)
x_2 = x_2.merge(
    df_1.reset_index(drop=True).reset_index(level=0)[["Fitness", "index"]], on="index"
)

plt.close()
plt.scatter(x.one, x.two, c=x.Fitness, cmap="Greens", alpha=0.4)
plt.scatter(
    x_2.one,
    x_2.two,
    c=x_2.Fitness,
    cmap="binary",
    alpha=0.1,
    s=(x_2.Fitness ** 4) / 10000000,
)
