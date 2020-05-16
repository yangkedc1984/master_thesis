# Playground
from run_HAR_model import *
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
from sklearn import metrics


df_input = load_data()
har_model_heteroskedasticity = HARModel(
    df=df_input,
    future=1,
    lags=[4, 20],
    feature="RV",
    semi_variance=True,
    jump_detect=True,
    log_transformation=False,
    period_train=list(
        [
            pd.to_datetime("20030910", format="%Y%m%d"),
            pd.to_datetime("20091231", format="%Y%m%d"),  # 20091231
        ]
    ),
    period_test=list(
        [
            pd.to_datetime("20100101", format="%Y%m%d"),
            pd.to_datetime("20101231", format="%Y%m%d"),
        ]
    ),
)
har_model_heteroskedasticity.predict_values()

marker_sty = ["p", "s", "8"]
colors = ["darkgreen", "mediumseagreen", "forestgreen"]
periods = [1, 5, 20]
plt.close()
for i in range(3):
    har_model_heteroskedasticity = HARModel(
        df=df_input,
        future=periods[i],
        lags=[4, 20],
        feature="RV",
        semi_variance=True,
        jump_detect=True,
        log_transformation=False,
        period_train=list(
            [
                pd.to_datetime("20030910", format="%Y%m%d"),
                pd.to_datetime("20091231", format="%Y%m%d"),  # 20091231
            ]
        ),
        period_test=list(
            [
                pd.to_datetime("20100101", format="%Y%m%d"),
                pd.to_datetime("20101231", format="%Y%m%d"),
            ]
        ),
    )
    har_model_heteroskedasticity.predict_values()
    plt.plot(
        np.log(har_model_heteroskedasticity.training_set.RV_t),
        har_model_heteroskedasticity.prediction_train
        - har_model_heteroskedasticity.training_set.future,
        marker_sty[i],
        # fillstyles="none",
        color=colors[i],
        alpha=0.3,
        label="HAR {}".format(periods[i]),
    )
plt.legend(loc="upper left")
plt.savefig("heteroskedasticity")


plt.plot(
    np.log(har_model_heteroskedasticity.prediction_train),
    har_model_heteroskedasticity.prediction_train
    - har_model_heteroskedasticity.training_set.future,
    marker_sty[i],
    # fillstyles="none",
    color=colors[i],
    alpha=0.3,
    label="HAR {}".format(periods[i]),
)

plt.close()
plt.acorr(
    har_model_heteroskedasticity.training_set.RV_t,
    maxlags=50,
    lw=2,
    color="mediumseagreen",
)


plt.close()
plt.plot(
    np.log(har_model_heteroskedasticity.prediction_train),
    har_model_heteroskedasticity.prediction_train
    - har_model_heteroskedasticity.training_set.future,
    ",",
    # fillstyles="none",
    color="mediumseagreen",
    alpha=0.3,
    label="HAR {}".format(periods[i]),
)


def print_results_log_har(
    prediction, future
):  # har_model_heteroskedasticity.training_set.future
    print(
        metrics.r2_score(future, prediction,),
        metrics.mean_squared_error(future, prediction,),
        metrics.mean_absolute_error(future, prediction,),
    )


print_results_log_har(
    har_model_heteroskedasticity.prediction_test,
    np.exp(har_model_heteroskedasticity.testing_set.future),
)

df_v = pd.read_csv(folder_structure.path_input + "/" + "DataFeatures.csv", index_col=0)
df_v.DATE = df_v.DATE.values
df_v.DATE = pd.to_datetime(df_v.DATE, format="%Y%m%d")

har_log_validation = HARModel(
    df=df_v,
    future=5,
    lags=[4, 20],
    feature="RV",
    semi_variance=True,
    jump_detect=True,
    log_transformation=False,
    period_train=list(
        [
            pd.to_datetime("20110101", format="%Y%m%d"),
            pd.to_datetime("20111231", format="%Y%m%d"),  # 20091231
        ]
    ),
    period_test=list(
        [
            pd.to_datetime("20110101", format="%Y%m%d"),
            pd.to_datetime("20111231", format="%Y%m%d"),
        ]
    ),
)
har_log_validation.make_testing_training_set()
prediction_validation = np.exp(
    har_model_heteroskedasticity.model.predict(
        har_log_validation.training_set[["RSV_plus", "RSV_minus", "RV_w", "RV_m"]]
    )
)
future_validation = np.exp(har_log_validation.training_set.future)
print_results_log_har(prediction=prediction_validation, future=future_validation)


# PLOTTING :: Check whether HAR log-transformation is correctly specified
pred_train_log = har_model_heteroskedasticity.prediction_train
future_train_log = np.exp(har_model_heteroskedasticity.training_set.future)
pred_train_normal = har_model_heteroskedasticity.prediction_train
future_train_normal = har_model_heteroskedasticity.training_set.future

plt.close()
plt.plot(
    har_model_heteroskedasticity.training_set.DATE,
    pred_train_log,
    lw=0.5,
    label="pred log",
)
plt.plot(
    har_model_heteroskedasticity.training_set.DATE,
    pred_train_normal,
    lw=0.5,
    label="pred normal",
)
plt.plot(
    har_model_heteroskedasticity.training_set.DATE,
    future_train_log,
    lw=0.5,
    label="future log",
)
plt.plot(
    har_model_heteroskedasticity.training_set.DATE,
    future_train_normal,
    lw=0.5,
    label="future normal",
)
plt.legend()


# colors_ = ["palegreen", "green", "black"]

# Breusch-Pagan Test
name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(
    har_model_heteroskedasticity.model.resid,
    har_model_heteroskedasticity.model.model.exog,
)
lzip(
    name, test
)  # p-value is very small, thus heteroskedasticity is present :: Null Hypothesis --> Homoskedasticity is present


for i in [1, 5, 20]:
    har_model_heteroskedasticity = HARModel(
        df=df_input,
        future=i,
        lags=[4, 20],
        feature="RV",
        semi_variance=True,
        jump_detect=True,
        log_transformation=False,
        period_train=list(
            [
                pd.to_datetime("20030910", format="%Y%m%d"),
                pd.to_datetime("20091231", format="%Y%m%d"),  # 20091231
            ]
        ),
        period_test=list(
            [
                pd.to_datetime("20100101", format="%Y%m%d"),
                pd.to_datetime("20101231", format="%Y%m%d"),
            ]
        ),
    )
    har_model_heteroskedasticity.predict_values()
    name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
    test = sms.het_breuschpagan(
        har_model_heteroskedasticity.model.resid,
        har_model_heteroskedasticity.model.model.exog,
    )
    print(lzip(name, test))

# Heteroskedastic-robust Standard errors
results_robust = har_model_heteroskedasticity.model.get_robustcov_results(
    cov_type="HAC", maxlags=20
)

print(results_robust.summary())  # you should use more data to obtain better results

print(results_robust.summary().as_latex())


df_v = pd.read_csv(
    folder_structure.path_input
    + "/"
    + "GeneticAlgorithm_20_hist40_True_new_modelafterGA2.csv",
    index_col=0,
)

df_fit_1 = df_v.iloc[df_v.Fitness.nlargest(50).index]

df_fit = df_v.iloc[df_v.Fitness.nlargest(10).index]


# Loss Fuctions: sigmoid, tanh(), ReLu, Elu


vector_help = np.arange(-4, 4, 0.1)


def sigmoid(input):
    return 1 / (1 + np.exp(-input))


def tanh(input):
    return (np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input))


def LeakyReLu(input):
    list_help = list([])
    for i in range(len(input)):
        list_help.append(max(0.05 * input[i], input[i]))
    return list_help


def ELU(input):
    list_help = list([])
    for i in range(len(input)):
        if input[i] < 0:
            list_help.append(0.5 * (np.exp(input[i]) - 1))
        else:
            list_help.append(input[i])
    return list_help


plt.close()
plt.plot(
    vector_help, sigmoid(vector_help), label="Sigmoid", color="darkgreen", alpha=0.7
)
plt.plot(
    vector_help[0:60],
    LeakyReLu(vector_help)[0:60],
    label="Leaky ReLU",
    color="mediumseagreen",
)
plt.plot(
    vector_help[0:60],
    ELU(vector_help)[0:60],
    "-.",
    label="ELU",
    color="darkgreen",
    alpha=0.7,
)
plt.plot(
    vector_help,
    tanh(vector_help),
    "-.",
    label="Hyperbolic Tangent",
    color="mediumseagreen",
)
plt.legend()
plt.savefig("plot")


# Linear Regression function with sklearn
from sklearn.linear_model import LinearRegression

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
X = np.dot(X, np.array([1, 2])) + 2
y = np.dot(X, np.array([1, 2])) + 3
import pandas as pd

y = pd.Series(y, name="k")
np.array(y).reshape(-1, 1)

reg = LinearRegression().fit(X, y)
reg.intercept_
reg.coef_


def mincer_zarno_alpha_beta(y_real, y_pred):
    y_pred = y_pred.reshape(-1, 1)
    reg = LinearRegression().fit(y_pred, y_real)
    beta = reg.coef_[0]
    alpha = reg.intercept_
    return alpha, beta


alpha, beta = mincer_zarno_alpha_beta(y, X)
alpha
beta

df_v = pd.read_csv(
    folder_structure.path_dashboard_deployment + "/" + "DashboardData.csv", index_col=0
)


df = df_v.copy()
df = df[
    [
        "A(1)",
        "period",
        "A(3)",
        "H(SV,L)",
        "H(RV,L)",
        "H(SV)",
        "H(RV)",
        "L(SV,20)",
        "L(SV,40)",
        "L(RV,20)",
        "L(RV,40)",
        "future",
    ]
].copy()

df.apply(lambda x: sum(x))
