print("run LSTM with RV 1 20 :: Patience 10 :: 10 iterations test_17")

from run_HAR_model import load_data
from config import folder_structure
from LSTM import *

"""
for RV 1 20: 20, 10, 20, 20 (15 iterations)

for RV 5 40: 40, 20, 20, 0
for SV 5 40: 40, 20, 20, 0
"""

df_input = load_data()


lstm_instance = TimeSeriesDataPreparationLSTM(
    df=df_input,
    future=1,
    lag=20,
    standard_scaler=False,
    min_max_scaler=True,
    log_transform=True,
    semi_variance=False,
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


# RV_1_hist20
# {'MAE': 0.05528467030995142, 'RSquared': 0.6182530245808628, 'MSE': 0.005204762214886979} test 0.01, 40, 20, 2, 0
# {'MAE': 0.05074624721994997, 'RSquared': 0.806114124429627, 'MSE': 0.0044134570510344934} train
# {'RSquared': 0.6065186150588646, 'MSE': 0.005364749890564172, 'MAE': 0.056405081571273066} test 0.01, 20, 10, 20, 20
# {'RSquared': 0.809105331234155, 'MSE': 0.004345367703506162, 'MAE': 0.05052816873485473} train

# SV_1_hist20 :: already exported 0.01, 40, 40, 0, 0

# SV_1_hist40 :: already exported  0.01, 40, 40, 0, 0

# RV_1_hist40:
# {'RSquared': 0.7508079392337841, 'MSE': 0.0057409411690484205, 'MAE': 0.05932607308649277}  0.01, 40, 40
# {'RSquared': 0.4714228674797769, 'MSE': 0.007206653789395419, 'MAE': 0.06499547086857925}
# {'MSE': 0.005676952496187735, 'MAE': 0.05796410945304809, 'RSquared': 0.5836198935622561}  40, 20, 2, 2 (Test)
# {'MSE': 0.004460371946643379, 'MAE': 0.05106668795563212, 'RSquared': 0.8063924982962133} (Train)
# {'MSE': 0.004448075305533245, 'RSquared': 0.8069262479460542, 'MAE': 0.051289956763540145} 0.01, 40, 40, 0, 0 (Train)
# {'MSE': 0.005192709216034299, 'RSquared': 0.6191370603286594, 'MAE': 0.05616397850639968} (Test)


# SV_5_hist20:
# {'RSquared': 0.8487892767074003, 'MSE': 0.004222057442180422, 'MAE': 0.05100818664132991}  0.005, 20, 20
# {'RSquared': 0.5501721215202318, 'MSE': 0.006624525669466637, 'MAE': 0.06152399637172061}
# {'MAE': 0.061450006169662084, 'RSquared': 0.5550299015668898, 'MSE': 0.0065529861092142455}  0.01, 2, 40, 2, 2
# {'MAE': 0.05201189886945681, 'RSquared': 0.8393594223436639, 'MSE': 0.0044853548190341135}


# SV_5_hist40:
# {'MSE': 0.003879958020320708, 'MAE': 0.04889480338661023, 'RSquared': 0.8627696158627143} Train 0.01, 20, 10, 20
# {'MSE': 0.006660488268822093, 'MAE': 0.059211571677829845, 'RSquared': 0.547730138413842} Test 0.01, 20, 10, 20


# RV_5_hist20: 0.005, 20, 20
# {'RSquared': 0.831359272236756, 'MSE': 0.004708732451003183, 'MAE': 0.05386661469167614}
# {'RSquared': 0.5066229141622732, 'MSE': 0.0072658661817592675, 'MAE': 0.06523030084253341}

# RV_5_hist40:
# {'MSE': 0.004105349451810179, 'MAE': 0.050546693782805935, 'RSquared': 0.8547977376716208} 0.01, 20, 20, 2
# {'MSE': 0.006660725305254758, 'MAE': 0.06266490710226297, 'RSquared': 0.5477140428319169}


# x.fitted_model.save(folder_structure.output_LSTM + "/" + "LSTM_hist40_sv_1_aftergeneticalgorithm.h5")
# x.fitted_model.save(folder_structure.output_LSTM + "/" + "new_run_model.h5")
# x.fitted_model.save(folder_structure.output_LSTM + "/" + "LSTM_SV_1.h5")
# x.fitted_model.save(folder_structure.output_LSTM + "/" + "LSTM_RV_1.h5")
# x.fitted_model.save(folder_structure.output_LSTM + "/" + "LSTM_SV_5.h5")
# x.fitted_model.save(folder_structure.output_LSTM + "/" + "LSTM_RV_5.h5")
# x.fitted_model.save(folder_structure.output_LSTM + "/" + "LSTM_SV_20.h5")
# x.fitted_model.save(folder_structure.output_LSTM + "/" + "LSTM_RV_20.h5")


# best model finder:
tf.keras.backend.clear_session()
best_model = TrainLSTM(
    lstm_instance.training_set,
    lstm_instance.testing_set,
    epochs=80,
    learning_rate=0.01,
    layer_one=40,
    layer_two=5,
    layer_three=0,
    layer_four=0,
    adam_optimizer=True,
)
best_model.make_accuracy_measures()
best_model.fitted_model.save(
    folder_structure.output_LSTM + "/" + "LSTM_False_1_20_v2.h5"
)

model_dict = {"model_first": best_model}
fitness_list = [best_model.fitness]

for i in range(5):
    print("Model {}".format(i))
    print("---------------------------------------------")
    tf.keras.backend.clear_session()
    x = TrainLSTM(
        lstm_instance.training_set,
        lstm_instance.testing_set,
        epochs=80,
        learning_rate=0.01,
        layer_one=40,
        layer_two=5,
        layer_three=0,
        layer_four=0,
        adam_optimizer=True,
    )
    x.make_accuracy_measures()
    print(x.train_accuracy)
    print(x.test_accuracy)
    print(x.fitness)

    model_dict["model_{}".format(i)] = x

    if x.fitness > best_model.fitness:
        del best_model
        best_model = x
        best_model.fitted_model.save(
            folder_structure.output_LSTM + "/" + "LSTM_False_1_20_v2.h5"
        )
        fitness_list.append(best_model.fitness)
    else:
        best_model = best_model
        fitness_list.append(best_model.fitness)


# print("BEST MODEL BEST MODEL")
# print(best_model.fitness)
# print(best_model.test_accuracy)
# print(best_model.train_accuracy)


dict_best_results = {}
for semi_value in [True, False]:
    lstm_instance = TimeSeriesDataPreparationLSTM(
        df=df_input,
        future=5,
        lag=40,
        standard_scaler=False,
        min_max_scaler=True,
        log_transform=True,
        semi_variance=semi_value,
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
    tf.keras.backend.clear_session()
    best_model = TrainLSTM(
        lstm_instance.training_set,
        lstm_instance.testing_set,
        epochs=80,
        learning_rate=0.01,
        layer_one=40,
        layer_two=20,
        layer_three=20,
        layer_four=0,
        adam_optimizer=True,
    )
    best_model.make_accuracy_measures()
    best_model.fitted_model.save(
        folder_structure.output_LSTM + "/" + "LSTM_{}_5_40_v2.h5".format(semi_value)
    )

    model_dict = {"model_first": best_model}
    fitness_list = [best_model.fitness]

    for i in range(15):
        print("Model {}".format(i))
        print("---------------------------------------------")
        tf.keras.backend.clear_session()
        x = TrainLSTM(
            lstm_instance.training_set,
            lstm_instance.testing_set,
            epochs=80,
            learning_rate=0.01,
            layer_one=40,
            layer_two=20,
            layer_three=20,
            layer_four=0,
            adam_optimizer=True,
        )
        x.make_accuracy_measures()
        print(x.train_accuracy)
        print(x.test_accuracy)
        print(x.fitness)

        model_dict["model_{}".format(i)] = x

        if x.fitness > best_model.fitness:
            del best_model
            best_model = x
            best_model.fitted_model.save(
                folder_structure.output_LSTM
                + "/"
                + "LSTM_{}_5_40_v2.h5".format(semi_value)
            )
            fitness_list.append(best_model.fitness)
        else:
            best_model = best_model
            fitness_list.append(best_model.fitness)

    dict_best_results["Model_{}".format(semi_value)] = best_model


for i in [True, False]:
    print(dict_best_results["Model_{}".format(i)].test_accuracy)
    print(dict_best_results["Model_{}".format(i)].train_accuracy)
    print(dict_best_results["Model_{}".format(i)].fitness)
