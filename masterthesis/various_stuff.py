# Playground
from LSTM import *
from run_HAR_model import *
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from statsmodels.stats import sandwich_covariance
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
    log_transformation=True,
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
print(har_model_heteroskedasticity.estimation_results)


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

# Heteroskedastic-robust Standard errors
results_robust = har_model_heteroskedasticity.model.get_robustcov_results(
    cov_type="HAC", maxlags=20
)

print(results_robust.summary())  # you should use more data to obtain better results

print(results_robust.summary().as_latex())


df_v = pd.read_csv(
    folder_structure.path_input + "/" + "GeneticAlgorithm_1_hist40_True.csv",
    index_col=0,
)

# the fittest individuals
df_fit = df_v.iloc[df_v.Fitness.nlargest(50).index]
df_fit.head()


# colorscheme: darkgreen mediumseagreen green

# solution finder ::

import plotly.graph_objects as go

df = pd.read_csv(
    "https://raw.githubusercontent.com/plotly/datasets/master/violin_data.csv"
)

fig = go.Figure()
df.columns
df.day.unique()

days = ["Thur", "Fri", "Sat", "Sun"]

day = "Thur"

df["day"][df["day"] == day]

df["total_bill"][df["day"] == day]


for day in days:
    fig.add_trace(
        go.Violin(
            x=df["day"][df["day"] == day],
            y=df["total_bill"][df["day"] == day],
            name=day,
            box_visible=True,
            meanline_visible=True,
        )
    )

fig.show()
