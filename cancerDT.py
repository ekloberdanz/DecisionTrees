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
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
import math
import sys

colNames = ["sampleCodeNumber", "clumpThickness", "uniformityOfCellSize", "uniformityOfCellShape", "marginalAdhesion",
"singleEpithelialCellSize ", "bareNuclei", "blandChromatin", "normalNucleoli ", "mitoses", "classification"]

data = sys.argv[1]
cancerDataSet = pd.read_csv(data, header=None, names=colNames)

cancerDataSet.replace(['?'],[None], inplace=True)
print("Number of rows dropped", cancerDataSet.shape[0] - cancerDataSet.dropna().shape[0])

cancerDataSet.dropna(how='any', inplace=True)


#cancerDataSet.to_csv("test.csv")

features = ["sampleCodeNumber", "clumpThickness", "uniformityOfCellSize", "uniformityOfCellShape", "marginalAdhesion",
"singleEpithelialCellSize ", "bareNuclei", "blandChromatin", "normalNucleoli ", "mitoses"]

X = cancerDataSet[features]
y = cancerDataSet.classification

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


tree = DecisionTreeClassifier()
tree = tree.fit(X_train, y_train)

print("Accuracy on training set:", tree.score(X_train, y_train))
print("Accuracy on testing set:", tree.score(X_test, y_test))

from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="cancerTree.dot", class_names=["benign", "malignant"],
                feature_names=features, impurity=False, filled=True)


with open("cancerTree.dot") as f:
    dot_graph = f.read()
src = graphviz.Source(dot_graph)
src.render('BreastCancerTree', view=True)


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


# 5 fold cross validation
scores = cross_val_score(tree, X, y, cv=5)
print("5-fold cross validation sccores: ", scores)
#print("average 5-fold cross validation score: ", scores.mean(), "+/-", scores.std()*2 )

# 95% confidence interval
z = 1.96
n = X.shape[0]
acc = scores.mean()
err = 1 - scores.mean()

CI = z * math.sqrt((acc*err)/n)

print("\nAccuracy based on 95 percent confidence interval")
print("average 5-fold cross validation score: ", scores.mean(), "+/-", CI)

