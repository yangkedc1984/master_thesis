from Result_Tables_Graphs import *

for i in [1, 5, 20]:
    result_instance = ResultOutput(forecast_period=i)
    result_instance.run_all(save_=True, save_plots=False)


result_instance = ResultOutput(forecast_period=5)
result_instance.run_all(save_=True, save_plots=False)


result_instance.prepare_lstm_data()
result_instance.prepare_har_data()

result_instance.unit_test_har.testing_set
result_instance.unit_test_lstm.testing_set


assert (
    round(
        sum(
            (
                result_instance.unit_test_lstm.back_transformation(
                    np.array(result_instance.unit_test_lstm.testing_set.future).reshape(
                        -1, 1
                    )
                ).reshape(result_instance.unit_test_lstm.testing_set.shape[0],)
                - result_instance.unit_test_har.testing_set.future
            )
        ),
        10,
    )
    == 0
), "Error: Unequal future realized volatility for LSTM- and HAR Model (unequal dependent variable in the models)"
