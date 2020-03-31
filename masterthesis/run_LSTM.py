from run_HAR_model import *
from LSTM import *

df_input = load_data()

lstm_instance = TimeSeriesDataPreparationLSTM(
    df=df_input,
    future=20,
    lag=40,
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

tf.keras.backend.clear_session()
x = TrainLSTM(
    lstm_instance.training_set,
    lstm_instance.testing_set,
    epochs=50,
    learning_rate=0.01,
    layer_one=40,
    layer_two=40,
    layer_three=0,
    layer_four=0,
    adam_optimizer=True,
)
x.make_accuracy_measures()

x.fitness
x.train_accuracy
x.test_accuracy

x.make_performance_plot(show_testing_sample=False)

# RV_1_hist20
# {'RSquared': 0.7986528505525539, 'MSE': 0.004583299293052354, 'MAE': 0.0517652077364555} train  0.01 40, 40
# {'RSquared': 0.5949158052675401, 'MSE': 0.005522943327256374, 'MAE': 0.057722057633606105} test


# SV_1_hist40 :: already exported  0.01, 40, 40, 0, 0

# RV_1_hist40:
# {'RSquared': 0.7508079392337841, 'MSE': 0.0057409411690484205, 'MAE': 0.05932607308649277}  0.01, 40, 40
# {'RSquared': 0.4714228674797769, 'MSE': 0.007206653789395419, 'MAE': 0.06499547086857925}


# SV_5_hist20:
# {'RSquared': 0.8487892767074003, 'MSE': 0.004222057442180422, 'MAE': 0.05100818664132991}  0.005, 20, 20
# {'RSquared': 0.5501721215202318, 'MSE': 0.006624525669466637, 'MAE': 0.06152399637172061}
# {'MAE': 0.061450006169662084, 'RSquared': 0.5550299015668898, 'MSE': 0.0065529861092142455}  0.01, 2, 40, 2, 2
# {'MAE': 0.05201189886945681, 'RSquared': 0.8393594223436639, 'MSE': 0.0044853548190341135}


# SV_5_hist40:
# {'RSquared': 0.8387270309102819, 'MSE': 0.004559721622979651, 'MAE': 0.052498139008333305} train  0.01, 20, 20
# {'RSquared': 0.46323555229140845, 'MSE': 0.00790482322776394, 'MAE': 0.06496865950888162} test
# {'MSE': 0.003879958020320708, 'MAE': 0.04889480338661023, 'RSquared': 0.8627696158627143} Train 0.01, 20, 10, 20
# {'MSE': 0.006660488268822093, 'MAE': 0.059211571677829845, 'RSquared': 0.547730138413842} Test 0.01, 20, 10, 20


# RV_5_hist20: 0.005, 20, 20
# {'RSquared': 0.831359272236756, 'MSE': 0.004708732451003183, 'MAE': 0.05386661469167614}
# {'RSquared': 0.5066229141622732, 'MSE': 0.0072658661817592675, 'MAE': 0.06523030084253341}

# RV_5_hist40: 0.01, 20, 20
# {'RSquared': 0.7810513240168452, 'MAE': 0.06017115199236228, 'MSE': 0.006190405111520981}
# {'RSquared': 0.40480945781503475, 'MAE': 0.07004520880662017, 'MSE': 0.008765252696772115}
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
    epochs=50,
    learning_rate=0.01,
    layer_one=40,
    layer_two=2,
    layer_three=2,
    layer_four=2,
    adam_optimizer=True,
)
best_model.make_accuracy_measures()

# best_model.fitted_model.save(
#     folder_structure.output_LSTM + "/" + "LSTM_RV_1_new_lift.h5"
# )
#
# best_model.fitness
# best_model.test_accuracy
# best_model.train_accuracy
#
# best_model.make_performance_plot(False)

model_dict = {"model_first": best_model}
fitness_list = [best_model.fitness]


for i in range(5):
    tf.keras.backend.clear_session()
    x = TrainLSTM(
        lstm_instance.training_set,
        lstm_instance.testing_set,
        epochs=50,
        learning_rate=0.01,
        layer_one=40,
        layer_two=40,
        layer_three=2,
        layer_four=2,
        adam_optimizer=True,
    )
    x.make_accuracy_measures()

    model_dict["model_{}".format(i)] = x

    if x.fitness > best_model.fitness:
        del best_model
        best_model = x
        fitness_list.append(best_model.fitness)
    else:
        best_model = best_model
        fitness_list.append(best_model.fitness)


# 2 40 2 2
# 40 40 0 0

plt.close()
plt.plot(fitness_list)

best_model.test_accuracy
best_model.train_accuracy
best_model.fitted_model
best_model.make_performance_plot(show_testing_sample=True)
x = best_model

# best_model.fitted_model.save(
#     folder_structure.output_LSTM + "/" + "LSTM_RV_1_new_lift.h5"
# )
#
# model_test = tf.keras.models.load_model(
#     folder_structure.output_LSTM + "/" + "LSTM_SV_1.h5"
# )
#
# model_dict["model_1"].fitted_model.save(
#     folder_structure.output_LSTM + "/" + "LSTM_RV_1_TRYTOEXPORT.h5"
# )

model_dict["model_4"].make_performance_plot(False)


for i in range(5):
    print("model_{}".format(i))
    print(model_dict["model_{}".format(i)].fitness)
    print(model_dict["model_{}".format(i)].train_accuracy)
    print(model_dict["model_{}".format(i)].test_accuracy)
    print("---------------------------------------------")
