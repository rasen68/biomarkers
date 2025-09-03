import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.naive_bayes import GaussianNB

#from https://www.geeksforgeeks.org/machine-learning/cross-validation-using-k-fold-with-scikit-learn/#kfold-with-scikitlearn
def cross_validation(reg_model, housing_prepared, housing_labels, cv):
    scores = cross_val_score(
      reg_model, housing_prepared,
      housing_labels,
      scoring="neg_mean_squared_error", cv=cv)
    rmse_scores = np.sqrt(-scores)
    print("Scores:", rmse_scores)
    print("Mean:", rmse_scores.mean())
    print("StandardDeviation:", rmse_scores.std())

X = [np.genfromtxt('../connectomes/rois_aal/'+i, delimiter=',').flatten() for i in sorted(os.listdir('../connectomes/rois_aal/'))]
df = pd.read_csv('../src/rois_aal/demographics.csv')
y_class = df['DSM_IV_TR'].values.tolist()
y_reg = df['ADOS_TOTAL'].values.tolist()

X_scaled = StandardScaler().fit_transform(X)
components = 70
pca = PCA(n_components=components, whiten=True).fit(X_scaled)
X_pca = pca.transform(X_scaled)

kf = KFold(n_splits=5, shuffle=True, random_state=0)
for i in [LinearRegression, SVR, GradientBoostingRegressor, MLPRegressor]:
    print(i.__name__)
    model = i()
    cross_validation(model, X_pca, y_reg, kf)

for i in [LogisticRegression, SVC, GradientBoostingClassifier, GaussianNB, MLPClassifier]:
    print(i.__name__)
    model = i()
    cross_validation(model, X_pca, y_class, kf)

''' Help choose # components
print("Cumulative:", cums := np.cumsum(pca.explained_variance_ratio_))
cums_1stdv = []
for i in range(1, len(cums)):
    cums_1stdv.append(cums[i] - cums[i-1])
cums_2nddv = []
for i in range(1, len(cums_1stdv)):
    cums_2nddv.append(cums_1stdv[i] - cums_1stdv[i-1])

print([float(i*10000) for i in cums_2nddv])
'''
