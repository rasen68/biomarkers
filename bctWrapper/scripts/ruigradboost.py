import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

X = [np.genfromtxt('../connectomes/rois_aal/'+i, delimiter=',').flatten() for i in sorted(os.listdir('../connectomes/rois_aal/'))]
df = pd.read_csv('../src/rois_aal/demographics.csv')
y_class = df['DSM_IV_TR'].values.tolist()
y_reg = df['ADOS_TOTAL'].values.tolist()

X_scaled = StandardScaler().fit_transform(X)
components = 70
pca = PCA(n_components=components, whiten=True).fit(X_scaled)
X_pca = pca.transform(X_scaled)

regmodel = GradientBoostingRegressor()
classmodel = GradientBoostingClassifier()

X_train, X_test, y_train, y_test = train_test_split(X_pca, y_class, test_size=0.2)

lrs = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]

for learning_rate in lrs:
    gb_clf = GradientBoostingClassifier(learning_rate=learning_rate, max_depth=4)
    gb_clf.fit(X_train, y_train)

    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test, y_test)))

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
