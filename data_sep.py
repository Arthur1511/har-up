import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy.ma import sort
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint
from sklearn.model_selection import train_test_split
from time import time
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

data = pd.read_csv("CompleteDataSet.csv", header=None, skiprows=2).dropna(axis=0)
# print(data.head())


ankle_data = data.iloc[:, 1:7]

sbj1_act1 = data.loc[(data[43] == 1)].loc[(data[44] == 1)].loc[(data[45] == 1)]

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
""" def report(results, n_top=3):
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

clf = RandomForestClassifier(n_estimators=20, n_jobs=4)

# specify parameters and distributions to sample from
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 6),
              "min_samples_split": sp_randint(2, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run randomized search
n_iter_search = 5
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=4, cv=5,
                                   iid=False)

start = time()
random_search.fit(X_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))

report(random_search.cv_results_)

y_pred = random_search.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(accuracy)

cm = confusion_matrix(y_test, y_pred)

print(cm)

plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, linewidths=.5, cmap=plt.cm.Blues, robust=True)
plt.show()
 """

sensors = {"Ankle": ankle_data, "Right Pocket": right_pocket_data, "Belt": belt_data, "Neck": neck_data,
           "Wrist": wrist_data}

features = ["Acelerometro X", "Acelerometro Y", "Acelerometro Z", "Giroscopio X", "Giroscopio Y",
            "Giroscopio Z"]

sensors_features = list(np.array([["Ankle " + feature for feature in features],
                                  ["Right Pocket " + feature for feature in features],
                                  ["Belt " + feature for feature in features],
                                  ["Neck " + feature for feature in features],
                                  ["Wrist " + feature for feature in features]]).flat)

atividades = ["Falling\nforward\nusing hands", "Falling\nforward\nusing knees", "Falling\nbackwards",
              "Falling\nsideward",
              "Falling sitting\nin empty chair", "Walking", "Standing", "Sitting", "Picking up\nan object", "Jumping",
              "Laying"]

total_data.columns = sensors_features

corr = total_data.corr()

plt.figure(dpi=300, figsize=(15, 10))
plt.title("Matriz de correlação - Todos os sensores", fontdict={'fontsize': 18})
sns.heatmap(corr.round(2), cmap="RdBu", xticklabels=sensors_features, yticklabels=sensors_features, vmin=-1, vmax=1,
            linecolor="black", linewidths=0.5)
# plt.xticks(rotation=45)
# plt.yticks(rotation=45)
# plt.savefig("img/tota_data.png", dpi=300, format="png")

plt.show()

for sensor in sensors:
    corr = sensors[sensor].corr()

    plt.figure(dpi=300, figsize=(15, 10))
    plt.title(sensor, fontdict={'fontsize': 18})
    sns.heatmap(corr, cmap="RdBu", vmin=-1, vmax=1, xticklabels=features, yticklabels=features, annot=True, square=True)
    plt.savefig("img/" + sensor + ".png", dpi=300)
    plt.show()

plt.figure(dpi=300, figsize=(15, 10))

sns.heatmap(cm_norm.round(3), xticklabels=atividades, yticklabels=atividades, cmap="RdBu", annot=True)
plt.xticks(rotation=0)
plt.show()
print(classification_report(y_test, y_pred, target_names=[a.replace("\n", " ") for a in atividades]))

clf_report = classification_report(y_test, y_pred, output_dict=True,
                                   target_names=[a.replace("\n", " ") for a in atividades])

clf_report = pd.DataFrame(data=clf_report).drop("support", axis=0).transpose()
clf_report.to_csv("classification_report.csv")
