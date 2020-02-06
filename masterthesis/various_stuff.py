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
lstm_instance.reshape_input_data()

model_ = tf.keras.models.load_model("LSTM_SV_1.h5")
predict_model = model_.predict(lstm_instance.test_matrix)
predict_model = lstm_instance.back_transformation(predict_model)

df_test = lstm_instance.testing_set.DATE.to_frame(name="DATE")
df_test["LSTM_pred"] = predict_model

df_test_2 = results["har_1_True"].testing_set[["future", "DATE"]]
df_test_2["HAR_pred"] = results["har_1_True"].prediction_test

df_test = df_test.merge(df_test_2, on="DATE")
df_test["indicator"] = df_test.DATE.map(lambda x: 100 * x.year + x.month)

for i in ["LSTM_pred", "HAR_pred", "future"]:
    df_test = df_test.merge(
        df_test.groupby("indicator")
        .apply(lambda x: x[i].mean())
        .to_frame(name="{}_average".format(i))
        .reset_index(level=0),
        on="indicator",
    )

df_test["performance_check"] = np.where(
    np.abs(df_test.LSTM_pred_average - df_test.future_average)
    > np.abs(df_test.HAR_pred_average - df_test.future_average),
    1,
    -1,  # if == 1, then HAR Model is superior
)

plt.close()
fig, axs = plt.subplots(2)
for i in ["LSTM_pred_average", "HAR_pred_average", "future_average"]:
    axs[0].plot(df_test.DATE, df_test[i], label=i, alpha=1, lw=0.5)
axs[0].legend()
axs[1].plot(df_test.DATE, df_test.performance_check, alpha=1, lw=0.5)


df_test["new"] = np.where(
    np.abs(df_test.future - df_test.LSTM_pred)
    > np.abs(df_test.future - df_test.HAR_pred),
    0,
    1,
)  # 1 == LSTM is better

plt.close()
plt.plot(df_test.DATE, df_test.new, ".", c="red", alpha=0.1)

# for 1 day ahead prediction out of sample we perform better with LSTM for all measurements
metrics.mean_absolute_error(
    df_test.future, df_test.HAR_pred
) > metrics.mean_absolute_error(df_test.future, df_test.LSTM_pred)

metrics.mean_squared_error(
    df_test.future, df_test.HAR_pred
) > metrics.mean_squared_error(df_test.future, df_test.LSTM_pred)

metrics.r2_score(df_test.future, df_test.HAR_pred) > metrics.r2_score(
    df_test.future, df_test.LSTM_pred
)

plt.close()
plt.plot(
    results["har_1_True"].training_set.DATE,
    results["har_1_True"].prediction_train,
    label="HAR",
)
plt.plot(lstm_instance.training_set.DATE, predict_model, label="LSTM")
plt.plot(
    lstm_instance.training_set.DATE,
    lstm_instance.back_transformation(
        np.array(lstm_instance.training_set.future).reshape(-1, 1)
    ),
    label="Realized Volatility LSTM",
)
plt.plot(
    results["har_1_True"].training_set.DATE,
    results["har_1_True"].training_set.future,
    label="Realized Vola HAR",
)
plt.legend()


"""
PCA starts here
"""

direc_links = [
    "InitialPopulation_all_scenarios_future_1_newfitness.csv",
    "InitialPopulation_all_scenarios_future_5_newfitness.csv",
    "InitialPopulation_all_scenarios_future_20_newfitness.csv",
]

direc_links_2 = [
    "InitialPopulation_sv_1_newfitness.csv",
    "InitialPopulation_sv_5_newfitness.csv",
    "InitialPopulation_sv_20_newfitness.csv",
]
titles_set = [
    "One Day",
    "One Week",
    "One Month",
]


df_1 = pd.read_csv(
    instance_path.path_input + "/" + "InitialPopulation_sv_1_newfitness.csv",
    index_col=0,
)
df_2 = df_1.iloc[df_1.Fitness.nlargest(40).index]

plt.close()
fig = plt.figure()
fig.suptitle("Hyperparameter Optimization using Genetic Algorithm", fontsize=16)
for i in range(3):
    initial_population_scenarios = pd.read_csv(
        instance_path.path_input + "/" + direc_links[i], index_col=0,
    )
    initial_population_scenarios = initial_population_scenarios.reset_index(level=0)

    df_1 = pd.read_csv(instance_path.path_input + "/" + direc_links_2[i], index_col=0,)
    df_2 = df_1.iloc[df_1.Fitness.nlargest(5).index]

    # print PCA
    pca = PCA(n_components=3)
    pca.fit(
        initial_population_scenarios[["LR", "Layer1", "Layer2", "Layer3", "Layer4"]]
    )
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
        df_1.reset_index(drop=True).reset_index(level=0)[["Fitness", "index"]],
        on="index",
    )

    pca_3 = PCA(n_components=3)
    pca_3.fit(df_1[["LR", "Layer1", "Layer2", "Layer3", "Layer4"]])

    x_3 = pd.DataFrame(
        pca.transform(df_2[["LR", "Layer1", "Layer2", "Layer3", "Layer4"]]),
        columns={"one", "two", "three"},
    ).reset_index(level=0)

    x_3 = x_3.merge(
        df_2.reset_index(drop=True).reset_index(level=0)[["Fitness", "index"]],
        on="index",
    )

    from mpl_toolkits.mplot3d import Axes3D

    axs = fig.add_subplot(1, 3, i + 1, projection="3d")
    axs.scatter(
        x_3.one, x_3.two, x_3.three, c="darkred", s=200, alpha=0.5,
    )
    axs.scatter(
        x.one, x.two, x.three, alpha=0.2, marker="o", s=100,
    )
    axs.scatter(
        x_2.one, x_2.two, x_2.three, c=x_2.Fitness, cmap="binary", alpha=1,
    )
    axs.view_init(elev=5, azim=50)
    axs.margins(x=-0.45)

    plt.rcParams["axes.edgecolor"] = "0.15"
    plt.rcParams["axes.linewidth"] = 1.25

    axs.set_title(titles_set[i])

# fig.savefig("test.png")  # produces some kind of error (but still works)
