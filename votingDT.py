import pandas as pd
import warnings
with warnings.catch_warnings():
	warnings.filterwarnings("ignore", category=FutureWarning)
	import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import graphviz
import pprint
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
import mglearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
import math
import sys

colNames = ["party", "handicappedInfants", "waterProjectCostSharing", "adoptionOfTheBudgetResolution", "physicianFeeFreeze",
"elSalvadorAid", "religiousGroupsInSchools", "antiSateliteTestBan", "aidToNicaraguanContras", "mxMissile", "immigration", "synfuelsCorporationCutback"
, "educationSpending", "superfundRightToSue", "crime", "dutyFreeExports", "exportAdministrationActSouthAfrica"]

data = sys.argv[1]
votingDataSet = pd.read_csv(data, header=None, names=colNames)

votingDataSet.replace(['n', 'y', '?'],[0,1, None], inplace=True)

print("Number of rows that contain missing values", votingDataSet.shape[0] - votingDataSet.dropna().shape[0])

#print(votingDataSet)
# Impute missing values
imp = SimpleImputer(strategy="most_frequent")
votingDataSet = pd.DataFrame(imp.fit_transform(votingDataSet),columns=colNames)
#print(votingDataSet)

#votingDataSet.dropna(how='any', inplace=True)

#votingDataSet.to_csv("test.csv")
features = ["handicappedInfants", "waterProjectCostSharing", "adoptionOfTheBudgetResolution", "physicianFeeFreeze",
"elSalvadorAid", "religiousGroupsInSchools", "antiSateliteTestBan", "aidToNicaraguanContras", "mxMissile", "immigration", "synfuelsCorporationCutback"
, "educationSpending", "superfundRightToSue", "crime", "dutyFreeExports", "exportAdministrationActSouthAfrica"]

X = votingDataSet[features]
y = votingDataSet.party

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


tree = DecisionTreeClassifier()
tree = tree.fit(X_train, y_train)

print("Accuracy on training set:", tree.score(X_train, y_train))
print("Accuracy on testing set:", tree.score(X_test, y_test))

from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="tree.dot", class_names=["republican", "democrat"],
                feature_names=features, impurity=False, filled=True)


with open("tree.dot") as f:
    dot_graph = f.read()
src = graphviz.Source(dot_graph)
src.render('CongressionalVotingTree', view=True)

print("Feature importances:")
print(tree.feature_importances_)

def plot_feature_importances_cancer(model):
    n_features = X.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), features)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    plt.show()

plot_feature_importances_cancer(tree)

#shuffle_split = ShuffleSplit(test_size=.8, train_size=.2, n_splits=5)
#scores = cross_val_score(tree, X, y, cv=shuffle_split)
#print("Cross-validation scores:\n{}".format(scores))

# 5 fold cross validation
from sklearn.model_selection import KFold # import KFold
kf = KFold(n_splits=5, random_state=42, shuffle=True) 

accuracyTrain = []
accuracyTest =[]
i = 1
for train_index, test_index in kf.split(X):
	X_train, X_test = X.iloc[train_index], X.iloc[test_index]
	y_train, y_test = y.iloc[train_index], y.iloc[test_index]

	tree = DecisionTreeClassifier()
	tree = tree.fit(X_train, y_train)

	export_graphviz(tree, out_file="tree"+str(i)+".dot", class_names=["republican", "democrat"],
                feature_names=features, impurity=False, filled=True)

	with open("tree"+str(i)+".dot") as f:
	    dot_graph = f.read()
	src = graphviz.Source(dot_graph)
	src.render('CongressionalVotingTree'+str(i)+"Fold", view=True)

	accuracyTrain.append(tree.score(X_train, y_train))
	accuracyTest.append(tree.score(X_test, y_test))

	print("\nk =", i)
	print("Accuracy on training set:", tree.score(X_train, y_train))
	print("Accuracy on testing set:", tree.score(X_test, y_test))

	i += 1

averageAccuracy = sum(accuracyTest)/len(accuracyTest)
# 95% confidence interval
z = 1.96
n = X.shape[0]
acc = averageAccuracy
err = 1 - averageAccuracy

CI = z * math.sqrt((acc*err)/n)

print("\nAccuracy based on 95 percent confidence interval:")
print("average accuracy on testing", averageAccuracy, "+/-", CI)
