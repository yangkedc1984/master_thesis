from config import *
from HAR_Model import *


def load_data():
    df_m = pd.read_csv(
        instance_path.path_input + "/" + "RealizedMeasures03_10.csv", index_col=0
    )
    df_m.DATE = df_m.DATE.values
    df_m.DATE = pd.to_datetime(df_m.DATE, format="%Y%m%d")

    return df_m


def estimate_and_predict_har_models(df_input, save=True):
    all_models = {"future": [1, 5, 20], "semi_variance": [True, False]}
    all_results = {}

    os.chdir(instance_path.output_path + "/" + instance_path.HARModel)

    for i in all_models["future"]:
        for k in all_models["semi_variance"]:
            all_results["har_{}_{}".format(i, k)] = HARModel(
                df=df_input,
                future=i,
                lags=[4, 20],
                feature="RV",
                semi_variance=k,
                jump_detec=True,
                period_train=list(
                    [
                        pd.to_datetime("20030910", format="%Y%m%d"),
                        pd.to_datetime("20080208", format="%Y%m%d"),
                    ]
                ),
                period_test=list(
                    [
                        pd.to_datetime("20080209", format="%Y%m%d"),
                        pd.to_datetime("20101231", format="%Y%m%d"),
                    ]
                ),
            )
            all_results["har_{}_{}".format(i, k)].run_complete_model()

            if save:
                estimation_results = open("har_{}_{}_estimation.txt".format(i, k), "a+")
                estimation_results.write(
                    all_results["har_{}_{}".format(i, k)].estimation_results
                )

                accuracy_results = open("har_{}_{}_accuracy.txt".format(i, k), "a+")
                accuracy_results.write("Train Accuracy:")
                accuracy_results.write(
                    str(all_results["har_{}_{}".format(i, k)].train_accuracy)
                )
                accuracy_results.write("Test Accuracy:")
                accuracy_results.write(
                    str(all_results["har_{}_{}".format(i, k)].test_accuracy)
                )

    return all_results


def run_all(save_output=True):
    df = load_data()
    res = estimate_and_predict_har_models(df_input=df, save=save_output)

    return res


results = run_all(save_output=False)

results["har_1_True"].prediction_test.head()
results["har_5_True"].prediction_test.head()
results["har_20_True"].prediction_test.head()

plt.hist(np.log(results["har_1_True"].df.RV), bins=100)  # makes it much better

results["har_1_True"].df.shape

from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler

scaler_ = MinMaxScaler()
results["har_1_True"].df.RV = scaler_.fit_transform(
    np.log(results["har_1_True"].df.RV.values).reshape(-1, 1)
)

df_view = results["har_1_True"].df

data = np.log(results["har_1_True"].df.RV.values)


ser = scaler_.fit_transform(data.reshape(-1, 1))
series_minmax = np.reshape(ser, (1791,))  # hard coded!!!

series_minmax_log_2 = np.where(series_minmax == 0, 0.5, series_minmax)

series = normalize(np.log(results["har_1_True"].df.RV.values.reshape(1, -1)))
df_view.RV = normalize(df_view.RV.values.reshape(-1, 1))


series = np.reshape(series, (1791,))  # hard coded!!!
ser = pd.Series(series)


scaler_2 = MinMaxScaler()
sca_2 = scaler_2.fit_transform(series_minmax_log_2.reshape(-1, 1))
sca_2 = np.reshape(sca_2, (1791,))
sca_2 = pd.Series(sca_2)


from scipy import stats

all_series = [series_minmax, series_minmax_log_2, sca_2]

color_hist = ["black", "navy", "green"]
color_density = ["pink", "c", "black"]

fig, axs = plt.subplots(3)
for i in range(len(all_series)):
    xt = plt.xticks()[0]
    xmin, xmax = min(xt), max(xt)
    lnspc = np.linspace(xmin, xmax, len(all_series[i]))
    m, s = stats.norm.fit(all_series[i])  # get mean and standard deviation
    pdf_g = stats.norm.pdf(lnspc, m, s)  # now get theoretical values in our interval
    axs[i].hist(all_series[i], bins=50, alpha=0.5, density=True, color=color_hist[i])
    axs[i].plot(lnspc, pdf_g, label="Norm", color=color_density[i])  # plot it


plt.close()
fig, axs = plt.subplots(2, 2)
axs[0, 0].hist(s, bins=50, color="black", alpha=0.5, density=True)
axs[0, 1].hist(series_minmax, bins=50, color="black", alpha=0.6, density=True)
axs[1, 0].hist(
    np.log(series_minmax_log_2), bins=50, color="black", alpha=0.7, density=True
)
axs[1, 1].hist(sca_2, bins=50, color="black", alpha=0.8, density=True)
axs[1, 1].plot(lnspc, pdf_g, label="Norm", color="pink")  # plot it


xt = plt.xticks()[0]
xmin, xmax = min(xt), max(xt)
lnspc = np.linspace(xmin, xmax, len(sca_2))

m, s = stats.norm.fit(sca_2)  # get mean and standard deviation
pdf_g = stats.norm.pdf(lnspc, m, s)  # now get theoretical values in our interval

plt.close()
plt.plot(lnspc, pdf_g, label="Norm", color="pink")  # plot it
plt.hist(sca_2, bins=100, color="black", alpha=0.6, density=True)

stats.kurtosis(s)
stats.kurtosis(series_minmax)
stats.kurtosis(series_minmax_log_2)
stats.kurtosis(sca_2)

stats.skew(s)
stats.skew(series_minmax)
stats.skew(series_minmax_log_2)
