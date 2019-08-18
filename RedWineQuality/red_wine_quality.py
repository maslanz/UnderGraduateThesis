import pandas as pd
#import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv('winequality-red.csv')

x = dataset.values[:, 0:11] #özniteliklerin ataması
y = dataset.values[:, 11] #etiket ataması

models = [
    ('Logistic Regression', LogisticRegression(random_state=0, multi_class='multinomial', solver='lbfgs', max_iter=7000)),
    ('Linear Discriminant Analysis', LinearDiscriminantAnalysis(n_components=8)),
    ('Support Vector Machine', SVC(random_state=0, gamma='scale')),
    ('Naive-Bayes Classifier', GaussianNB()),
    ('K-Nearest Neighborhood', KNeighborsClassifier(n_neighbors=19, weights='distance')),
    ('Basic Decision Tree', DecisionTreeClassifier(random_state=0)),
    ('Bagged Tree', BaggingClassifier(random_state=0)),
    ('Boosted Tree', GradientBoostingClassifier(random_state=0)),
    ('Random Forest', RandomForestClassifier(random_state=0, n_estimators=100))
]

accuracy_list = []
cv_score_list = []
i = 0

print("Model Name                     Accuracy               CV Score")
print("----------                     --------               --------")
for name, model in models:
    classifier = model
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=0, stratify=y)
    classifier.fit(x_train, y_train)
    test_result = classifier.predict(x_test)
    performances = accuracy_score(test_result, y_test)
    accuracy_list.append(performances)
    kfold = model_selection.KFold(n_splits=5)
    cv_results = model_selection.cross_val_score(classifier, x, y, cv=kfold)
    cv_score_list.append(cv_results.mean())
    print("%-30s %-22.4f %.4f" % (name, accuracy_list[i], cv_score_list[i]))
    i = i + 1

cm_list = []
j = 0

print(" ")
print("Confusion Matrices of Classification Models")
print("-------------------------------------------")
for name, model in models:
    classifier = model
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=0, stratify=y)
    classifier.fit(x_train, y_train)
    test_result = classifier.predict(x_test)
    cm = str(confusion_matrix(y_test, test_result))
    cm_list.append(cm)
    print(name + ": ")
    print("-----------------------------")
    print(cm_list[j])
    print(" ")
    j = j + 1