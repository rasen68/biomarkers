import os
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

'''
left = [4, 4, 10, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 15, 24, 25, 27, 32, 33, 38, 42, 42, 59, 69, 70, 71, 71, 80, 83, 86, 91, 96, 96]
right = [42, 44, 101, 101, 19, 55, 69, 70, 71, 76, 77, 91, 101, 42, 69, 101, 101, 27, 101, 83, 101, 101, 42, 96, 101, 101, 101, 91, 91, 101, 101, 102, 96, 101, 98, 100]
X_big = [np.genfromtxt('../connectomes/all/' + i, delimiter=',') for i in sorted(os.listdir('../connectomes/all'))]
X = [[big[b[0]][b[1]] for b in zip(left, right)] for big in X_big]
print(len(X[0]))
'''
X = [np.genfromtxt('../connectomes/rois_aal_NYU/' + i, delimiter=',').flatten() for i in sorted(os.listdir('../connectomes/rois_aal_NYU'))]
df = pd.read_csv('../src/rois_aal/demographics.csv')
df = df[df['SITE_ID'] == 'NYU']
y_class = df['DSM_IV_TR'].values.tolist()
y_reg = df['ADOS_TOTAL'].values.tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_final = X_scaled

selecter = SelectKBest(f_classif, k=84)
X_kbest = selecter.fit_transform(X_scaled, y_class)
print(selecter.get_feature_names_out())

pca = PCA(n_components=72, whiten=True).fit(X_scaled)
X_pca = pca.transform(X_scaled)
X_final = X_kbest

folds = 5
lr = 0.01
training_scores = []
val_scores = []
coefs = []
for i in range(folds):
    X_train, X_test, y_train, y_test = train_test_split(X_final, y_class, test_size=0.2)
    model = SVC()#GradientBoostingClassifier(learning_rate=lr, max_depth=3)
    model.fit(X_train, y_train)
    training_scores.append(model.score(X_train, y_train))
    val_scores.append(model.score(X_test, y_test))
    print(y_test[:10])
    print([round(float(y), 1) for y in model.predict(X_test)[:20]])
    #print(model.coef_[0])
print("Mean accuracy score (training): {0:.3f}".format(float(np.mean(training_scores))))
print("{0:.3f}".format(float(np.mean(val_scores))))
print("Validation scores:", [round(float(i), 3) for i in val_scores])

'''
lrs = [0.01]#, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25]
# best for svc - 72
ks = [40]#, 50, 60, 70, 80, 90, 100, 110, 120]
ns = []#[40, 45, 50, 55, 60, 65, 68, 70, 71]

lr = 0.1
for n in ns: #lr in lrs:
    pca = PCA(n_components=n, whiten=True).fit(X_kbest)
    X_pca = pca.transform(X_kbest)
    X_final = X_pca
    #selecter = SelectKBest(f_classif, k=k)
    #X_kbest = selecter.fit_transform(X_scaled, y_class)
    #X_final = X_kbest
    training_scores = []
    val_scores = []
    for i in range(folds):
        X_train, X_test, y_train, y_test = train_test_split(X_final, y_class, test_size=0.2)
        model = SVC()#GradientBoostingClassifier(learning_rate=lr, max_depth=3)
        model.fit(X_train, y_train)
        training_scores.append(model.score(X_train, y_train))
        val_scores.append(model.score(X_test, y_test))
    print("principal components:", n)#"k best: ", k)#"Learning rate: ", learning_rate)

    print("Mean accuracy score (training): {0:.3f}".format(float(np.mean(training_scores))))
    print("Mean accuracy score (validation): {0:.3f}".format(float(np.mean(val_scores))))
    print("Validation scores:", [round(float(i), 3) for i in val_scores])
'''
