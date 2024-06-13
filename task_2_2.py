import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score

input_file = 'income_data.txt'
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line[:-1].split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

X = np.array(X)
print(X)

label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

classifier_poly = SVC(kernel='poly', degree=8, max_iter=5000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
classifier_poly.fit(X_train, y_train)
f1_poly = cross_val_score(classifier_poly, X, y, scoring='f1_weighted', cv=3)
print(f1_poly)
print("F1 score with polynomial kernel: " + str(round(100*f1_poly.mean(), 2)) + "%")

classifier_rbf = SVC(kernel='rbf', degree=8, max_iter=5000)
classifier_rbf.fit(X_train, y_train)
f1_rbf = cross_val_score(classifier_rbf, X, y, scoring='f1_weighted', cv=3)
print("F1 score with RBF kernel: " + str(round(100*f1_rbf.mean(), 2)) + "%")

classifier_sigmoid = SVC(kernel='sigmoid', degree=8, max_iter=5000)
classifier_sigmoid.fit(X_train, y_train)
f1_sigmoid = cross_val_score(classifier_sigmoid, X, y, scoring='f1_weighted', cv=3)
print("F1 score with sigmoid kernel: " + str(round(100*f1_sigmoid.mean(), 2)) + "%")
