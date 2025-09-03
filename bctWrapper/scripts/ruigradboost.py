import os
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

X = [np.genfromtxt('../connectomes/rois_aal/'+i, delimiter=',').flatten() for i in sorted(os.listdir('../connectomes/rois_aal/'))]
df = pd.read_csv('../src/rois_aal/demographics.csv')
y_class = df['DSM_IV_TR'].values.tolist()
y_reg = df['ADOS_TOTAL'].values.tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

threshold = VarianceThreshold(threshold=0.5)
X_threshold = threshold.fit_transform(X_scaled)

selecter = SelectKBest(f_classif, k=170)
X_kbest = selecter.fit_transform(X_threshold, y_class)

pca = PCA(n_components=70, whiten=True).fit(X_kbest)
X_pca = pca.transform(X_kbest)

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_pca, y_class, test_size=0.2)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_pca, y_class, test_size=0.2)
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X_pca, y_class, test_size=0.2)
X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split(X_pca, y_class, test_size=0.2)
X_train_5, X_test_5, y_train_5, y_test_5 = train_test_split(X_pca, y_class, test_size=0.2)

lrs = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]

for learning_rate in lrs:
    gb_clf_1 = GradientBoostingClassifier(learning_rate=learning_rate, max_depth=3)
    gb_clf_1.fit(X_train_1, y_train_1)
    gb_clf_2 = GradientBoostingClassifier(learning_rate=learning_rate, max_depth=3)
    gb_clf_2.fit(X_train_2, y_train_2)
    gb_clf_3 = GradientBoostingClassifier(learning_rate=learning_rate, max_depth=3)
    gb_clf_3.fit(X_train_3, y_train_3)
    gb_clf_4 = GradientBoostingClassifier(learning_rate=learning_rate, max_depth=3)
    gb_clf_4.fit(X_train_4, y_train_4)
    gb_clf_5 = GradientBoostingClassifier(learning_rate=learning_rate, max_depth=3)
    gb_clf_5.fit(X_train_5, y_train_5)
    print("Learning rate: ", learning_rate)

    training_scores = [gb_clf_1.score(X_train_1, y_train_1),
                       gb_clf_2.score(X_train_2, y_train_2),
                       gb_clf_3.score(X_train_3, y_train_3),
                       gb_clf_4.score(X_train_4, y_train_4),
                       gb_clf_5.score(X_train_5, y_train_5)]
    val_scores = [gb_clf_1.score(X_test_1, y_test_1),
                  gb_clf_2.score(X_test_2, y_test_2),
                  gb_clf_3.score(X_test_3, y_test_3),
                  gb_clf_4.score(X_test_4, y_test_4),
                  gb_clf_5.score(X_test_5, y_test_5)]
    print("Mean accuracy score (training): {0:.3f}".format(float(np.mean(training_scores))))
    print("Mean accuracy score (validation): {0:.3f}".format(float(np.mean(val_scores))))
    print("Validation scores:", [round(float(i), 3) for i in val_scores])
