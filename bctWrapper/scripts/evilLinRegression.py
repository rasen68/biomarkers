import os
import sys

import helpers 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVR
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier

from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kf = KFold(n_splits=10, shuffle=True, random_state=0)

# from https://www.geeksforgeeks.org/machine-learning/cross-validation-using-k-fold-with-scikit-learn/#kfold-with-scikitlearn
def cross_validation(reg_model, housing_prepared, housing_labels, cv):
    scores = cross_val_score(
      reg_model, housing_prepared,
      housing_labels,
      scoring="neg_mean_squared_error", cv=cv)
    rmse_scores = np.sqrt(-scores)
    print("Scores:", rmse_scores)
    print("Mean:", rmse_scores.mean())
    print("StandardDeviation:", rmse_scores.std())

def trainModel(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    pca = PCA(n_components=10)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.fit_transform(X_test_scaled)

    for i in [LinearRegression, SVR, GradientBoostingRegressor]:
        print(i.__name__)
        model = i()
        cross_validation(model, X_train_scaled, y_train, kf)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        r2 = r2_score(y_test, y_pred)
        print(f"\nR-squared: {r2:.4f}")

        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean squared error: {mse:.4f}")

        rmse = mse ** 0.5
        print(f"Root mean squared error: {rmse:.4f}\n")

        if i == LinearRegression:
            np.savetxt('weights_orig.csv', np.reshape(model.coef_, (116, 116)), delimiter=',')

    for i in [LinearRegression, SVR, GradientBoostingRegressor]:
        print(i.__name__)
        model = i()
        cross_validation(model, X_train_pca, y_train, kf)
        model.fit(X_train_pca, y_train)
        y_pred = model.predict(X_test_scaled)

        r2 = r2_score(y_test, y_pred)
        print(f"\nR-squared: {r2:.4f}")

        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean squared error: {mse:.4f}")

        rmse = mse ** 0.5
        print(f"Root mean squared error: {rmse:.4f}\n")

        if i == LinearRegression:
            np.savetxt('weights_pca.csv', np.reshape(model.coef_, (116, 116)), delimiter=',')

def trainTree(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    for i in [LogisticRegression, GradientBoostingClassifier, DecisionTreeClassifier, GaussianNB, MLPClassifier]:
        print(i.__name__)
        model = i()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        print([int(i) for i in y_pred])
        print(y_test)

        true_pos = sum(y_pred[i] == 1 and y_test[i] == 1 for i in range(len(y_test)))
        predicted_pos = sum(y_pred[i] == 1 for i in range(len(y_pred)))
        actual = sum(y_test[i] == 1 for i in range(len(y_test)))

        precision = true_pos / predicted_pos
        print(f"Precision: {precision:.4f}")

        recall = true_pos / actual
        print(f"Recall: {recall:.4f}")

        F = 2*precision*recall / (precision+recall)
        print(f"F-measure: {F:.4f}")

        accuracy = sum(y_pred[i] == y_test[i] for i in range(len(y_test))) / len(y_test)
        print(f"Accuracy: {accuracy:.4f}")

        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean squared error: {mse:.4f}")

        rmse = mse ** 0.5
        print(f"Root mean squared error: {rmse:.4f}\n")

def plotCorrelation(X, y, feature, score):
    try:
        r, p = pearsonr(X, y)
        output_path = f"./scatterPlots/{feature}_vs_{score}.png"
        title = f"{feature} vs {score}"
        helpers.drawCorrelationPlot(X, y, r, p, feature, score, title, output_path)
        print(f"Saved plot: {output_path}")
    except Exception as e:
        print(f"Error plotting")

def getColumnFromCSV(file_path, col_name):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    df = pd.read_csv(file_path)
    
    if col_name not in df.columns:
        raise ValueError(f"The column '{col_name}' does not exist in the CSV file.")
    
    return df[col_name]

#if len(sys.argv) != 3:
#    sys.exit()

if __name__ == "__main__":
    # RUN THIS COMMAND 
    # python evilLinRegression.py ..\src\rois_aal\scores.csv ..\src\rois_aal\demographics.csv            

    #featuresList = pd.read_csv(sys.argv[1]).columns.tolist()[1:]

    varianceFile = pd.read_csv(sys.argv[1])

    variance = pd.DataFrame(varianceFile)[["variance"]]
    ADOS = getColumnFromCSV(sys.argv[1], "ADOS_TOTAL")

    r2 = trainModel(variance, ADOS)
    print(r2)

    '''
    X = [np.genfromtxt('../connectomes/rois_aal/'+i, delimiter=',').flatten() for i in sorted(os.listdir('../connectomes/rois_aal/'))] #pd.read_csv(sys.argv[1], header=0, index_col=False)
    score = getColumnFromCSV(sys.argv[2], 'DSM_IV_TR')
    y = score.values.tolist()

    trainTree(X, y)

    y2 = getColumnFromCSV(sys.argv[2], 'ADOS_TOTAL').values.tolist()

    #trainModel(X, y2)

    X2 = [[np.var(row)] for row in np.genfromtxt('../features/strength_interHemisphere.txt')]

    #trainModel(X2, y2)
    #trainTree(X2, y)

    '''

    demographics = ["FIQ", "PIQ", "AGE_AT_SCAN", "SEX", "VIQ", "ADOS_MODULE", "ADOS_TOTAL", "ADOS_COMM", "ADOS_SOCIAL", "ADOS_STEREO_BEHAV"]

    featuresList = pd.read_csv(sys.argv[1]).columns.tolist()[1:]
    graphTheoryMeasures = pd.read_csv(sys.argv[1])

    posCorr = []
    posCorr2 = []
    for demo in demographics:
        numSubjects = 171
        score = getColumnFromCSV(sys.argv[2], demo)
        X = pd.DataFrame(graphTheoryMeasures)
        y = score.values.tolist()[:numSubjects]

        r2 = trainModel(X, y)

        for feature in featuresList:
            if (feature == "AGE_AT_SCAN" or feature == "SEX"): continue
            print(feature)
            # Create features X and target y.
            X = pd.DataFrame(graphTheoryMeasures)[[feature]][:numSubjects]
            #X = pd.DataFrame(graphTheoryMeasures)[[feature, "AGE_AT_SCAN", "SEX"]][:170]
            y = score.values.tolist()[:numSubjects]
        
            #print(featuresList)
            #print("GT X: \n", X)
            #print("GT y: \n", y)

            r2 = trainModel(X, y)
            plotCorrelation(X.values.flatten().tolist(), y, feature, demo)
            if r2 > .015:
                posCorr.append(f'GT Measure: {feature}, demographic: {demo}, r2: {r2:.4f}')
                posCorr2.append(f'{r2:.4f}')
    print(posCorr)
    print(posCorr2)
        '''
