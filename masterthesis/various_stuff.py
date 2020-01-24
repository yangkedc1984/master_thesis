from run_HAR_model import *
from LSTM import *

from sklearn.decomposition import PCA

df_m = pd.read_csv(
    instance_path.path_input + "/" + "InitialPopulation_sv_1.csv", index_col=0
)
df_m = df_m.iloc[40:]
df_m.Fitness.nlargest(10).index


df_p = df_m[["LR", "BS", "Layer1", "Layer2", "Layer3", "Layer4"]]

pca = PCA(n_components=2)
pca.fit(df_p)
x = pd.DataFrame(pca.transform(df_p), columns={"one", "two"}).reset_index(level=0)
x = x.merge(df_m["Generation"].to_frame().reset_index(level=0), on="index")
x.Generation = np.where(x.Generation.isna(), 1, 0)
x.Generation.iloc[df_m.Fitness.nlargest(3).index] = 2

plt.close()
plt.scatter(x.one, x.two, c=x.Generation)

plt.plot(df_m.Fitness)


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
    epochs=20,
    learning_rate=0.06,
    layer_one=28,
    layer_two=16,
    layer_three=20,
    layer_four=20,
)
x.make_accuracy_measures()
x.make_performance_plot(show_testing_sample=True)

x.make_performance_plot(show_testing_sample=False)


# check whether new parents are selected each time, or whether they stay the same... (would be a mistake)
# how to code that nicely


# in general, how can this be nicely visualized...  (PCA is quite nice)


learning_rates = [0.0001, 0.05, 0.1]
layer_one = [10, 25, 40]
layer_two = [10, 25, 40]
layer_three = [0, 5, 10, 20]
layer_four = [0, 5, 10, 20]

number_of_models = (
    len(learning_rates)
    * len(layer_one)
    * len(layer_two)
    * len(layer_three)
    * len(layer_four)
)
number_of_models


# All Scenarios
dict_help = {}
for i in range(len(learning_rates)):
    for j in range(len(layer_one)):
        for k in range(len(layer_two)):
            for c in range(len(layer_three)):
                for f in range(len(layer_four)):
                    dict_help["{}{}{}{}{}".format(i, j, k, c, f)] = [
                        learning_rates[i],
                        layer_one[j],
                        layer_two[k],
                        layer_three[c],
                        layer_four[f],
                    ]
initial_population_scenarios = np.transpose(pd.DataFrame(dict_help)).reset_index(
    drop=True
)


# print PCA
pca = PCA(n_components=3)
pca.fit(initial_population_scenarios)
x = pd.DataFrame(
    pca.transform(initial_population_scenarios), columns={"one", "two", "three"}
).reset_index(level=0)

pca_2 = PCA(n_components=3)
pca_2.fit(df_m[["LR", "Layer1", "Layer2", "Layer3", "Layer4"]])
x_2 = pd.DataFrame(
    pca.transform(df_m[["LR", "Layer1", "Layer2", "Layer3", "Layer4"]]),
    columns={"one", "two", "three"},
).reset_index(level=0)
x_2 = x_2.merge(
    df_m.reset_index(drop=True).reset_index(level=0)[["Fitness", "index"]], on="index"
)

plt.close()
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    x.one, x.two, x.three, c=x.three, cmap="viridis", marker="^", s=50, alpha=0.3
)
ax.scatter(x_2.one, x_2.two, x_2.three, c=x_2.Fitness, cmap="binary")
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_zlabel("PC 3")


# analysis with two components
pca = PCA(n_components=2)
pca.fit(initial_population_scenarios)
x = pd.DataFrame(
    pca.transform(initial_population_scenarios), columns={"one", "two"}
).reset_index(level=0)

pca_2 = PCA(n_components=2)
pca_2.fit(df_m[["LR", "Layer1", "Layer2", "Layer3", "Layer4"]])
x_2 = pd.DataFrame(
    pca.transform(df_m[["LR", "Layer1", "Layer2", "Layer3", "Layer4"]]),
    columns={"one", "two"},
).reset_index(level=0)
x_2 = x_2.merge(
    df_m.reset_index(drop=True).reset_index(level=0)[["Fitness", "index"]], on="index"
)


plt.close()
plt.scatter(x.one, x.two, color="darkgreen", alpha=0.2)
plt.scatter(x_2.one, x_2.two, c=x_2.Fitness, cmap="binary", alpha=0.7)
