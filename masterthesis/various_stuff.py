# Playground
from LSTM import *
from run_HAR_model import *
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from statsmodels.stats import sandwich_covariance
import statsmodels.stats.api as sms
from statsmodels.compat import lzip


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
har_model_heteroskedasticity.predict_values()
print(har_model_heteroskedasticity.estimation_results)


results_robust = har_model_heteroskedasticity.model.get_robustcov_results(
    cov_type="HAC", maxlags=20
)
print(results_robust.summary())  # you should use more data to obtain better results


residuals = har_model_heteroskedasticity.prediction_train - np.exp(
    har_model_heteroskedasticity.training_set.future
)

plt.close()
plt.plot(
    har_model_heteroskedasticity.training_set.RV_t,
    residuals,
    "o",
    color="black",
    alpha=0.2,
)

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


import sklearn

sklearn.metrics.r2_score(
    np.exp(har_model_heteroskedasticity.training_set.future),
    har_model_heteroskedasticity.prediction_train,
)
sklearn.metrics.mean_squared_error(
    np.exp(har_model_heteroskedasticity.training_set.future),
    har_model_heteroskedasticity.prediction_train,
)
sklearn.metrics.mean_absolute_error(
    np.exp(har_model_heteroskedasticity.training_set.future),
    har_model_heteroskedasticity.prediction_train,
)

# Plotting the shit
plt.close()
plt.plot(
    har_model_heteroskedasticity.training_set.DATE,
    np.exp(har_model_heteroskedasticity.training_set.future),
    label="Realized Volatility",
    lw=0.6,
)
plt.plot(
    har_model_heteroskedasticity.training_set.DATE,
    har_model_heteroskedasticity.prediction_train,
    label="Predicted Volatility",
    lw=0.6,
)
plt.legend()

plt.plot()


dft = train_set[["RV_t", "RSV_plus", "RSV_minus"]].head()
dft["sum"] = dft.RSV_plus + dft.RSV_minus
dft

# construct weighted least squares
input_regression = train_set[["RV_t", "RV_w", "RV_m"]]
input_regression = train_set[["RSV_plus", "RSV_minus", "RV_w", "RV_m"]]

model = LinearRegression()
model.fit(input_regression, train_set.future)
prediction_train_sklearn = model.predict(input_regression)

model_wls_sklearn = LinearRegression()
model_wls_sklearn.fit(
    input_regression, train_set.future, sample_weight=1 / prediction_train_sklearn,
)

print("Regular OLS:", model.intercept_, model.coef_)
print("Weighted OLS:", model_wls_sklearn.intercept_, model_wls_sklearn.coef_)


periods_ = [1, 5, 20]
plt.close()
fig, axs = plt.subplots(3)
for i in range(3):
    har_model_heteroskedasticity = HARModel(
        df=df_input,
        future=periods_[i],
        lags=[4, 20],
        feature="RV",
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
    har_model_heteroskedasticity.make_testing_training_set()
    har_model_heteroskedasticity.predict_values()
    train_set = har_model_heteroskedasticity.training_set
    prediction_train = har_model_heteroskedasticity.prediction_train
    res = train_set.future - prediction_train
    axs[i].plot(
        prediction_train,
        res,
        ".",
        color="black",
        alpha=0.5,
        label="HAR Model {}".format(periods_[i]),
    )
    axs[i].legend()


df_v = pd.read_csv(
    folder_structure.path_input
    + "/"
    + "GeneticAlgorithm_1_hist_40_GeneticAlgorithm50generations_real_1.csv",
    index_col=0,
)

# fittest for lr == 0.001
df_001 = df_v.loc[df_v.LR == 0.001]
df_001_fittest = df_001.iloc[df_001.Fitness.nlargest(20).index]

# the fittest individuals
df_fit = df_v.iloc[df_v.Fitness.nlargest(20).index]
df_t = df_fit.copy()
df_t = df_t.iloc[0:10]
df_fit.head()

# NETWORK ANALYSIS:
# best network with all four layers
plt.close()
plt.plot(df_v.Fitness, ".")

df_34 = df_v.loc[(df_v.Layer3 != 0) & (df_v.Layer4 != 0)].reset_index(
    drop=True
)  # & (df_v.LR == 0.001)
df_34_fitness = df_34.iloc[df_34.Fitness.nlargest(20).index]


xlist = np.linspace(-3.0, 3.0, 20)
xlist = xlist.reshape(20, 1)
# ELU

data_ = np.linspace(-2.5, 2.5, 41)
import math

_mse = data_ ** 2
_mae = np.abs(data_)

_logcosh = list()
for i in data_:
    _logcosh.append(np.log(math.cosh(i)))

plt.close()
plt.plot(data_, _mse, color="darkgreen", label="Mean Squared Error")
plt.plot(data_, _mae, color="mediumseagreen", label="Mean Absolut Error")
plt.plot(
    data_, _logcosh, color="green", label="Hyperbolic Cosine Logarithm ", alpha=0.5
)
plt.ylabel("Loss")
plt.legend()
plt.savefig("Loss_Functions.png")


def elu(alpha, *args):
    if args < 0:
        return alpha * (np.exp(args) - 1)
    else:
        return args


y = elu(0.01, xlist)

y

# contour plot with plt
def f_hyper(x, y):
    return x ** 2 + y ** 2 + x * y


import matplotlib.mlab as mlab
import numpy as np
import matplotlib.pyplot as plt

xlist = np.linspace(-5.0, 5.0, 20)
ylist = np.linspace(-5.0, 5.0, 20)
X, Y = np.meshgrid(xlist, ylist)
Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
z = f_hyper(X, Y)
z = 10.0 * (Z1 - Z2)

levels = np.arange(0, 70, 2)

plt.close()
plt.figure()
cp = plt.contour(X, Y, z, levels=levels, cmap="viridis")
plt.clabel(cp)
plt.plot(X, Y, ".", color="gray")
plt.show()


delta = 0.25
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
# difference of Gaussians
Z = 10.0 * (Z2 - Z1)

plt.close()
plt.figure()
levels = np.arange(-1.0, 1.5, 0.1)
CS = plt.contour(X, Y, Z, levels=levels, cmap="summer")
plt.clabel(CS, inline=1, fontsize=10)
plt.plot(X, Y, ".", color="black", alpha=0.3)
plt.axis("off")
plt.show()
plt.savefig("GridSearch.png")


# self implemented gradient descent: for the function: x**2 + x*y + y**2
def f_grad(x, y):
    grad = np.array([[2 * x + y], [2 * y + x]])
    return grad


theta = np.array([[4], [3]])
grad = f_grad(int(theta[0]), int(theta[1]))
eta = 0.001

theta_list0 = list(theta[0])
theta_list1 = list(theta[1])
n_iterations = 5000

for i in range(n_iterations):
    grad = f_grad(float(theta[0]), float(theta[1]))
    print(grad)
    theta = theta - eta * grad
    print(theta)
    theta_list0.append(theta[0])
    theta_list1.append(theta[1])

theta  # solution for the function

plt.close()
plt.plot(theta_list0)
plt.plot(theta_list1)
