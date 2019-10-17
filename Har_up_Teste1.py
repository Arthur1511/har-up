# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# !apt-get install build-essential swig
# !pip install auto-sklearn

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint
from sklearn.model_selection import train_test_split
from time import time
from sklearn.metrics import confusion_matrix, accuracy_score
import autosklearn.classification

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# Any results you write to the current directory are saved as output.




data = pd.read_csv("CompleteDataSet.csv", header=None, skiprows=2)
# print(data.head())

ankle_data = data.iloc[:, 1:7]

# sbj1_act1 = data.loc[(data[43] == 1)].loc[(data[44] == 1)].loc[(data[45] == 1)]

# plt.figure(dpi=300)
# plt.plot([i / 10 for i in range(len(sbj1_act1[0]))], sbj1_act1.iloc[:, 8], label='x')
# plt.plot([i / 10 for i in range(len(sbj1_act1[0]))], sbj1_act1.iloc[:, 9], label='y')
# plt.plot([i / 10 for i in range(len(sbj1_act1[0]))], sbj1_act1.iloc[:, 10], label='z')
# plt.title("Falling forward using hands - Ankle Accelerometer")
# plt.xlabel("Time (s)")
# plt.ylabel("g")
# plt.legend(loc="lower right")
# plt.show()

# sns.lineplot([i / 10 for i in range(len(sbj1_act1[0]))], sbj1_act1.iloc[:, 8], legend='full')
# sns.lineplot([i / 10 for i in range(len(sbj1_act1[0]))], sbj1_act1.iloc[:, 9], legend='full')
# sns.lineplot([i / 10 for i in range(len(sbj1_act1[0]))], sbj1_act1.iloc[:, 10], legend='full')
# plt.show()

# print(ankle_data.head())

right_pocket_data = data.iloc[:, 8:14]

# print(right_pocket_data.head())

belt_data = data.iloc[:, 15:21]

neck_data = data.iloc[:, 22:28]

wrist_data = data.iloc[:, 29:35]

eeg_data = data.iloc[:, 36]

total_data = pd.concat([ankle_data, right_pocket_data, belt_data, neck_data, wrist_data], axis=1)

activity = data.iloc[:, -3]

# print(activity)


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


X_train, X_test, y_train, y_test = train_test_split(ankle_data, activity, test_size=0.3, random_state=42)

"""
automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=120,
        per_run_time_limit=30,
        tmp_folder='/tmp/autosklearn_cv_example_tmp',
        output_folder='/tmp/autosklearn_cv_example_out',
        delete_tmp_folder_after_terminate=False,
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds': 5}, n_jobs=4, seed=5)
"""
automl = autosklearn.classification.AutoSklearnClassifier(resampling_strategy='cv', resampling_strategy_arguments={'folds': 5}, n_jobs=4, seed=5, include_estimators=["random_forest", "libsvm_svc", "adaboost"], ensemble_size==0)

start = time()
automl.fit(X_train.copy(), y_train.copy(), dataset_name='har_up')
print("AutoML took %.2f seconds to fit" % (time() - start))

automl.refit(X_train.copy(), y_train.copy())
print("AutoML took %.2f seconds to refit" % (time() - start))

print(automl.show_models())

predictions = automl.predict(X_test)
print("Accuracy score", accuracy_score(y_test, predictions))


# clf = RandomForestClassifier(n_estimators=20, n_jobs=4)

# # specify parameters and distributions to sample from
# param_dist = {"max_depth": [3, None],
#               "max_features": sp_randint(1, 6),
#               "min_samples_split": sp_randint(2, 11),
#               "bootstrap": [True, False],
#               "criterion": ["gini", "entropy"]}

# # run randomized search
# n_iter_search = 20
# random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=4, cv=5,
#                                    iid=False)

# start = time()
# random_search.fit(X_train, y_train)
# print("RandomizedSearchCV took %.2f seconds for %d candidates"
#       " parameter settings." % ((time() - start), n_iter_search))

# report(random_search.cv_results_)

# y_pred = random_search.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)

# print(accuracy)

# cm = confusion_matrix(y_test, y_pred)

# print(cm)

# # use a full grid over all parameters
# param_grid = {"max_depth": [3, None],
#               "max_features": [1, 3, 6],
#               "min_samples_split": [2, 3, 10],
#               "bootstrap": [True, False],
#               "criterion": ["gini", "entropy"]}

# # run grid search
# grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5, iid=False)
# start = time()
# grid_search.fit(X_train, y_train)

# print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
#       % (time() - start, len(grid_search.cv_results_['params'])))
# report(grid_search.cv_results_)

# y_pred = grid_search.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)

# print(accuracy)

# cm = confusion_matrix(y_test, y_pred)

# print(cm)