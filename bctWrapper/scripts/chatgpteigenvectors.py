import numpy as np
import pandas as pd
from scipy.linalg import eigh
import networkx as nx
from tqdm import tqdm
import os

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

def make_adjacency_from_fc(fc, T=0.2):
    A = np.zeros_like(fc)
    A[fc >= T] = 1
    A[fc <= -T] = -1
    np.fill_diagonal(A, 0)
    return A

def laplacian_spectrum(adj):
    # adj: n x n adjacency as above
    deg = np.sum(adj, axis=1)  # degree (signed degrees here)
    D = np.diag(deg)
    L = D - adj
    # compute all eigenvalues (sorted)
    w, _ = eigh(L)
    # return as 1D array (length n)
    return np.sort(w)   # ascending

def build_features(FCs, T=0.2):
    n = len(FCs)
    spectra = []
    assort = []
    clustering = []
    avgdeg = []
    for i in range(n):
        print(i)
        fc = FCs[i]
        A = make_adjacency_from_fc(fc, T=T)
        w = laplacian_spectrum(A)
        if i == 1: print(len(w))
        spectra.append(w)
        # convert to binary adjacency for centralities where necessary
        # Mostafa binarizes positive edges for centralities (they do ai,j=1 if >0 else 0 for assort/clustering)
        Apos = (A > 0).astype(int)
        G = nx.from_numpy_array(Apos)
        # assortativity (degree assortativity)
        try:
            assort.append(nx.degree_assortativity_coefficient(G))
        except Exception:
            assort.append(0.0)
        # avg clustering coefficient
        clustering.append(nx.average_clustering(G))
        # average degree
        avgdeg.append(np.mean([d for _, d in G.degree()]))
    X = np.hstack([np.array(spectra),
                   np.array(assort)[:,None],
                   np.array(clustering)[:,None],
                   np.array(avgdeg)[:,None]])
    return X

FCs = [np.genfromtxt('../connectomes/all/' + i, delimiter=',') for i in sorted(os.listdir('../connectomes/all'))]

# labels: array shape (n,)
X = build_features(FCs, T=0.2)
df = pd.read_csv('../src/rois_aal/demographics.csv')
y = df['DSM_IV_TR'].values

# normalize numeric features to [0,1] as they did
scaler = MinMaxScaler()
Xs = scaler.fit_transform(X)

# LDA baseline
lda = LinearDiscriminantAnalysis()
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Sequential backward feature selection
sfs = SequentialFeatureSelector(lda,
                                n_features_to_select=64,  # paper ended with ~64; you can tune
                                direction='backward',
                                cv=cv,
                                n_jobs=-1)

sfs.fit(Xs, y)
mask = sfs.get_support()
X_sel = Xs[:, mask]
print("Selected features:", X_sel.shape)

from sklearn.model_selection import cross_validate

models = {
    'LDA': LinearDiscriminantAnalysis(),
}

for name, model in models.items():
    scores = cross_validate(model, X_sel, y, cv=cv,
                            scoring=['accuracy','roc_auc'],
                            n_jobs=-1, return_train_score=False)
    print(name, "ACC mean±std:", np.mean(scores['test_accuracy']), np.std(scores['test_accuracy']),
          "AUC mean±std:", np.mean(scores['test_roc_auc']), np.std(scores['test_roc_auc']))

