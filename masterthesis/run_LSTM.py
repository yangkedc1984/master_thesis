print("run LSTM with SV 20 20 :: Patience 10 :: 25 iterations test_31")

from run_HAR_model import load_data
from config import folder_structure
from LSTM import *

df_input = load_data()

lstm_instance = TimeSeriesDataPreparationLSTM(
    df=df_input,
    future=20,
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
# FINAL MODEL: 0.1, 40, 20, 20


# RV_5_hist20: 0.005, 20, 20
# {'RSquared': 0.831359272236756, 'MSE': 0.004708732451003183, 'MAE': 0.05386661469167614}
# {'RSquared': 0.5066229141622732, 'MSE': 0.0072658661817592675, 'MAE': 0.06523030084253341}

# RV_5_hist40:
# {'MSE': 0.004105349451810179, 'MAE': 0.050546693782805935, 'RSquared': 0.8547977376716208} 0.01, 20, 20, 2
# {'MSE': 0.006660725305254758, 'MAE': 0.06266490710226297, 'RSquared': 0.5477140428319169}
# FINAL MODEL: 0.1, 40, 20, 20


# RV_20_hist20: (final)
# {'MAE': 0.09018634713556153, 'RSquared': -0.0656223543838459, 'MSE': 0.01757234940605542} test (0.01, 40, 40, 0, 0)
# {'MAE': 0.07902062658021333, 'RSquared': 0.7591043832309334, 'MSE': 0.011178067692667937} train
# {'MSE': 0.017521419369207633, 'RSquared': -0.06253385525828836, 'MAE': 0.09527160851334837} test
# {'MSE': 0.009267059240890644, 'RSquared': 0.8002880271574908, 'MAE': 0.07400625765971035} train
# {'MSE': 0.017194817481669518, 'RSquared': -0.04272806467773904, 'MAE': 0.09352973324244554}
# {'MSE': 0.008978191271567812, 'RSquared': 0.8065133453026381, 'MAE': 0.07368819730698471}


# RV_20_hist40: (KING)
# {'RSquared': -0.03985491165125854, 'MAE': 0.09369400709612313, 'MSE': 0.017147438549845614} testing (0.01, 40, 40, .)
# {'RSquared': 0.8248749023517244, 'MAE': 0.0687051622773893, 'MSE': 0.008224533474769433} training


# SV_20_hist20:
# {'RSquared': -0.06205225604145648, 'MAE': 0.09138088202210212, 'MSE': 0.017513477691110287} 0.01, 40, 40 test
# {'RSquared': 0.8045135985720511, 'MAE': 0.07203521321282834, 'MSE': 0.009070983762450375} train


# SV_20_hist40:
# {'RSquared': -0.032467918223159975, 'MSE': 0.01702562538681954, 'MAE': 0.09370524668568629}
# {'RSquared': 0.7948573371357978, 'MSE': 0.009634271275146298, 'MAE': 0.07726463519088164}
# {'MAE': 0.09651640319653036, 'MSE': 0.01646446446715425, 'RSquared': 0.0015620004054707204}
# {'MAE': 0.07898189255631353, 'MSE': 0.010130623190260129, 'RSquared': 0.7842885093878318}


# best model finder:
tf.keras.backend.clear_session()
best_model = TrainLSTM(
    lstm_instance.training_set,
    lstm_instance.testing_set,
    epochs=80,
    learning_rate=0.01,
    layer_one=40,
    layer_two=40,
    layer_three=0,
    layer_four=0,
    adam_optimizer=True,
)
best_model.make_accuracy_measures()

best_model.fitted_model.save(
    folder_structure.output_LSTM + "/" + "LSTM_True_20_20_v22.h5"
)

model_dict = {"model_first": best_model}
fitness_list = [best_model.fitness]

for i in range(25):
    print("Model {}".format(i))
    print("---------------------------------------------")
    tf.keras.backend.clear_session()
    x = TrainLSTM(
        lstm_instance.training_set,
        lstm_instance.testing_set,
        epochs=80,
        learning_rate=0.01,
        layer_one=40,
        layer_two=40,
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
            folder_structure.output_LSTM + "/" + "LSTM_True_20_20_v22.h5"
        )
        fitness_list.append(best_model.fitness)
    else:
        best_model = best_model
        fitness_list.append(best_model.fitness)


print("BEST MODEL BEST MODEL")
print(best_model.fitness)
print(best_model.test_accuracy)
print(best_model.train_accuracy)
print("SV 20 20")
