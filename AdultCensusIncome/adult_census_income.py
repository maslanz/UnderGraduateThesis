import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv("adult_census_income.csv")
label_encoder = preprocessing.LabelEncoder()
categorical = list(dataset.select_dtypes(include=['object']).columns.values)
for cat in categorical:
    dataset[cat] = label_encoder.fit_transform(dataset[cat].astype('category'))

x = dataset.values[:, 0:14] #özniteliklerin ataması
y = dataset.values[:, 14]   #etiket ataması
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=0, stratify=y)

models = [
    ('Logistic Regression', LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000)),
    ('Linear Discriminant Analysis', LinearDiscriminantAnalysis()),
    ('Support Vector Machine', SVC(random_state=0, gamma='scale', probability=True)),
    ('Naive-Bayes Classifier', GaussianNB()),
    ('K-Nearest Neighborhood', KNeighborsClassifier(n_neighbors=11, weights='distance')),
    ('Basic Decision Tree', DecisionTreeClassifier(random_state=0)),
    ('Bagged Tree', BaggingClassifier(random_state=0)),
    ('Boosted Tree', GradientBoostingClassifier(random_state=0)),
    ('Random Forest', RandomForestClassifier(random_state=0, n_estimators=100))
]

accuracy_list = []
cv_score_list = []
roc_auc_list = []
i = 0

print("Model Name                     Accuracy       CV Score       ROC-AUC Score")
print("----------                     --------       --------       -------------")
for name, model in models:
    classifier = model #sınıflandırıcı modeli ataması
    classifier.fit(x_train, y_train) #x öznitelikleri ve y etiketine göre öğrenme sağlama
    test_result = classifier.predict(x_test) #x_test verisinin model üzerinde sınıflandırılması
    performances = accuracy_score(test_result, y_test)
    accuracy_list.append(performances)
    kfold = model_selection.KFold(n_splits=5)
    cv_results = model_selection.cross_val_score(classifier, x, y, cv=kfold)
    cv_score_list.append(cv_results.mean())
    roc_result = classifier.predict_proba(x_test)
    roc_auc_proba = roc_auc_score(y_test, roc_result[:, 1])
    roc_auc_list.append(roc_auc_proba)
    print("%-30s %-14.4f %-14.4f %.4f" % (name, accuracy_list[i], cv_score_list[i], roc_auc_list[i]))
    i = i + 1

k = 0

print(" ")

for name, model in models:
    classifier = model
    classifier.fit(x_train, y_train)
    roc_result = classifier.predict_proba(x_test)
    fpr, tpr, thresholds = roc_curve(y_test, roc_result[:, 1])
    plt.plot(fpr, tpr, label = name + ", roc-auc=%.5s" % (str(roc_auc_list[k])))
    plt.legend(loc=4)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    k = k + 1

k = 0

for name, model in models:
    classifier = model
    classifier.fit(x_train, y_train)
    roc_result = classifier.predict_proba(x_test)
    fpr, tpr, thresholds = roc_curve(y_test, roc_result[:, 1])
    plt.plot(fpr, tpr, label = name + ", roc-auc=%.5s" % (str(roc_auc_list[k])))
    k = k + 1
    
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='small')
plt.show()
    
cm_list = []
j = 0

print(" ")
print("Confusion Matrices of Classification Models")
print("-------------------------------------------")
for name, model in models:
    classifier = model
    classifier.fit(x_train, y_train)
    test_result = classifier.predict(x_test)
    cm = (confusion_matrix(y_test, test_result))
    cm_list.append(cm)
    print(name + ": ")
    print("-----------------------------")
    print(str(cm_list[j]))
    print("Sensitivity = %.2f" % (cm_list[j][1][1] / (cm_list[j][1][0] + cm_list[j][1][1])))
    print("Specifity = %.2f" % (cm_list[j][0][0] / (cm_list[j][0][0] + cm_list[j][0][1])))
    print(" ")
    j = j + 1