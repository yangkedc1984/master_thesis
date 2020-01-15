import pandas as pd
import statsmodels.formula.api as smf
from sklearn import metrics
import matplotlib.pyplot as plt

plt.style.use('seaborn')


class HARModel:

    def __init__(self,
                 df,
                 future=1,
                 lags=[4, 20],
                 feature='RV_s',
                 semi_variance=False,
                 period_train=list([pd.to_datetime('20030910', format='%Y%m%d'),
                                    pd.to_datetime('20081001', format='%Y%m%d')]),
                 period_test=list([pd.to_datetime('20090910', format='%Y%m%d'),
                                   pd.to_datetime('20100301', format='%Y%m%d')])
                 ):
        self.df = df
        self.future = future
        self.lags = lags
        self.feature = feature
        self.semi_variance = semi_variance
        self.period_train = period_train
        self.period_test = period_test
        self.training_set = None  # data frames
        self.testing_set = None  # data frames
        self.prediction_train = None  # vector (or excel export)
        self.prediction_test = None  # vector (or excel export)
        self.model = None  # stats model instance
        self.estimation_results = None  # table
        self.test_accuracy = None  # dictionary
        self.train_accuracy = None
        self.output_df = None  # DataFrame (data frame which contains all the features needed for the regression)

    def lag_average(self):

        df = self.df[['DATE', self.feature]]
        df['RV_t'] = df[self.feature].shift(-1)
        df['RV_w'] = df[self.feature].rolling(window=self.lags[0]).mean()

        rw20 = self.df[self.feature].rolling(window=self.lags[1]).mean()

        df['RV_m'] = list(
            ((self.lags[1] * rw20) - (self.lags[0] * df.RV_w)) /
            (self.lags[1] - self.lags[0])
        )

        df['DATE'] = df.DATE.shift(-1)

        df = df.dropna().reset_index(drop=True)

        df.drop([self.feature], axis=1, inplace=True)

        assert round(df.RV_t[0:self.lags[0]].mean() - df.RV_w[self.lags[0]], 12) == 0, 'Error'

        self.output_df = df

    def future_average(self):

        self.lag_average()
        df = self.output_df.copy()
        df_help = pd.DataFrame()

        for i in range(self.future):
            df_help[str(i)] = df.RV_t.shift(-(1 + i))
        df_help = df_help.dropna()

        df['future'] = df_help.mean(axis=1)

        df = df.dropna().reset_index(drop=True)

        self.output_df = df

        assert (self.output_df.RV_t[1:(self.future + 1)].mean()
                - self.output_df.future[0] == 0), 'Error'

    def generate_complete_data_set(self):

        if self.semi_variance:
            self.future_average()
            df = self.output_df.copy()
            df = df.merge(self.df[['DATE', 'RSV_s_plus', 'RSV_s_minus']], on='DATE')  # RSV_s_... is hard coded!

        else:
            self.future_average()
            df = self.output_df

        self.output_df = df

        assert (self.output_df.RV_t.iloc[1:(self.future + 1)].mean()
                - self.output_df.future[0] == 0), 'Error'

    def make_testing_training_set(self):
        self.generate_complete_data_set()
        df = self.output_df.copy()

        df_train = (df.loc[(df.DATE >= self.period_train[0])
                           & (df.DATE <= self.period_train[1])].reset_index(drop=True))
        df_test = (df.loc[(df.DATE >= self.period_test[0]) & (df.DATE <= self.period_test[1])].reset_index(drop=True))

        self.training_set = df_train
        self.testing_set = df_test

    def estimate_model(self):
        self.make_testing_training_set()

        if self.semi_variance:
            result = smf.ols(formula='future ~ RSV_s_plus + RSV_s_minus + RV_w + RV_m', data=self.training_set).fit()
        else:
            result = smf.ols(formula='future ~ RV_t + RV_w + RV_m', data=self.training_set).fit()

        self.model = result
        self.estimation_results = result.summary().as_latex()

    def predict_values(self):
        self.estimate_model()
        if self.semi_variance:
            self.prediction_train = self.model.predict(self.training_set[['RSV_s_plus', 'RSV_s_minus', 'RV_w', 'RV_m']])
            self.prediction_test = self.model.predict(self.testing_set[['RSV_s_plus', 'RSV_s_minus', 'RV_w', 'RV_m']])
        else:
            self.prediction_train = self.model.predict(self.training_set[['RV_t', 'RV_w', 'RV_m']])
            self.prediction_test = self.model.predict(self.testing_set[['RV_t', 'RV_w', 'RV_m']])

    def make_accuracy_measures(self):
        """
        Function that reports the accuracy measures for the out-of-sample and the in-sample prediction.
        Accuracy measures are: RMSE, MAE, MAPE and the R-Squared, Beta and Alpha of the
        Mincer-Zarnowitz Regression (R-Squared should be as high as possible, Beta equal to one and alpha equal to zero)

        :return:
        """
        self.predict_values()

        test_accuracy = {'MSE': metrics.mean_squared_error(self.testing_set['future'], self.prediction_test),
                         'MAE': metrics.mean_absolute_error(self.testing_set['future'], self.prediction_test),
                         'RSquared': metrics.r2_score(self.testing_set['future'], self.prediction_test)}
        train_accuracy = {'MSE': metrics.mean_squared_error(self.training_set['future'], self.prediction_train),
                          'MAE': metrics.mean_absolute_error(self.training_set['future'], self.prediction_train),
                          'RSquared': metrics.r2_score(self.training_set['future'], self.prediction_train)}

        self.test_accuracy = test_accuracy
        self.train_accuracy = train_accuracy

    def run_complete_model(self):
        self.make_accuracy_measures()

    def make_graph(self):
        self.predict_values()

        plt.figure()
        plt.plot(self.testing_set.DATE, self.testing_set.future, label='Realized Volatility')
        plt.plot(self.testing_set.DATE, self.prediction_test, label='Predicted Volatility')
        plt.legend()
        plt.show()
