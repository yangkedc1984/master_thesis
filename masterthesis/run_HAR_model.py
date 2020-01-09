from config import instance_path
from HAR_Model import *


def load_data():
    df_m = pd.read_csv(instance_path.path_input + '/' + 'RealizedMeasures03_10.csv',
                       index_col=0)
    df_m.DATE = df_m.DATE.values
    df_m.DATE = pd.to_datetime(df_m.DATE, format='%Y%m%d')

    return df_m


def estimate_and_predict_har_models(df_input):
    all_models = {'future': [1, 5, 20], 'semi_variance': [True, False]}
    all_results = {}

    for i in all_models['future']:
        for k in all_models['semi_variance']:
            all_results['har_{}_{}'.format(i, k)] = HARModel(
                df=df_input,
                future=i,
                lags=[4, 20],
                feature='RV_s',
                semi_variance=k,
                period_train=list([pd.to_datetime('20030910', format='%Y%m%d'),
                                   pd.to_datetime('20081001', format='%Y%m%d')]),
                period_test=list([pd.to_datetime('20090910', format='%Y%m%d'),
                                  pd.to_datetime('20100301', format='%Y%m%d')]))
            all_results['har_{}_{}'.format(i, k)].run_complete_model()
            # all_results['har_{}_{}'.format(i, k)].make_graph()

    return all_results


def save_models():
    x = None

    return x


def run_all():
    df = load_data()
    res = estimate_and_predict_har_models(df_input=df)
    save_models()

    return res


results = run_all()


results['har_1_True'].prediction_test.head()
results['har_5_True'].prediction_test.head()
results['har_20_True'].prediction_test.head()
