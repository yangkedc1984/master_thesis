from run_HAR_model import *

plt.style.use("seaborn")


results["har_1_True"].make_graph()

print(results["har_20_True"].estimation_results)


for i in [1, 5, 20]:
    plt.plot(
        results["har_{}_True".format(i)].training_set.DATE,
        results["har_{}_True".format(i)].prediction_train,
        label="{} day".format(i),
        alpha=0.5,
    )
plt.legend()
plt.close()

_training_set = results["har_20_True"].training_set
_training_set = _training_set.iloc[0:30]


_training_set.RV_t.iloc[1:21].mean() - _training_set.future[0]  # test passed
