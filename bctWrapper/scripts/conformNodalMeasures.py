import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import helpers

import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
# Import train_test_split.
from sklearn.model_selection import train_test_split
# Import StandardScaler.
from sklearn.preprocessing import StandardScaler
# Import LinearRegression.
from sklearn.linear_model import LinearRegression
# Import metrics.
from sklearn.metrics import mean_squared_error, r2_score

def trainModel(X, y):
    # Split the dataset into training (80%) and testing (20%) sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instantiate StandardScaler.
    scaler = StandardScaler()

    # Fit and transform training data.
    X_train_scaled = scaler.fit_transform(X_train)

    # Also transform test data.
    X_test_scaled = scaler.transform(X_test)

    # Instantiate linear regression model.
    model = LinearRegression()

    # Fit the model to the training data.
    model.fit(X_train_scaled, y_train)

    # Make predictions on the testing data.
    y_pred = model.predict(X_test_scaled)

    # Calculate and print R^2 score.
    r2 = r2_score(y_test, y_pred)
    print(f"\nR-squared: {r2:.4f}")

    # Calculate and print MSE.
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean squared error: {mse:.4f}")

    # Calculate and print RMSE.
    rmse = mse ** 0.5
    print(f"Root mean squared error: {rmse:.4f}\n")

    return r2

def plotCorrelation(X, y, measure, node):
    try:
        r, p = pearsonr(X, y)
        output_path = f"./scatterPlots/{measure}_{node}_vs_ADOS.png"
        title = f"{measure} Node{node} vs ADOS"
        helpers.drawCorrelationPlot(X, y, r, p, f'{measure} Node{node}', "ADOS", title, output_path)
        print(f"Saved plot: {output_path}")
    except Exception as e:
        print(f"Error plotting")

if __name__ == "__main__":

    nodes_of_interest = [37, 38, 39, 40, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 49, 50, 51, 52, 53, 54, 41, 42, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108]
    graphTheoryMeaures = ["clustering_coefficient", "clustering_coefficient_neg", "clustering_coefficient_zhang", "clustering_coefficient_zhang_neg", "degree", "degree_interHemisphere", "degree_interHemisphere_neg", "degree_intraHemisphere", "degree_intraHemisphere_neg", "degree_neg", "eccentricity", "eigenvector_centrality", "modularity_nodal", "modularity_nodal_neg", "node_betweenness_centrality", "strength_interHemisphere", "strength_interHemisphere_neg", "strength_intraHemisphere", "strength_noSelf", "strength_noSelf", "strength_selfConnections_nodal"]

    features = sys.argv[1]
    subjects_file = sys.argv[2]
    ados_file = sys.argv[3]

    meaningfulResults = []

    for measure in graphTheoryMeaures:
        featurePath = f'{features}{measure}.txt'
        if os.path.exists(featurePath):
            for node in nodes_of_interest:
                nodeName = f'Node{node}'

                print(f'{node} vs {measure}:\n')
                # Load files, limit to first 70 rows
                subjects = pd.read_csv(subjects_file, header=None, names=["SubjectID"], nrows=70)
                ados = pd.read_csv(ados_file, header=None, names=["ADOS"], nrows=70)
                nodal = pd.read_csv(featurePath, header=None, delim_whitespace=True, nrows=70)

                # Label nodal measure columns
                nodal.columns = [f"Node{i+1}" for i in range(nodal.shape[1])]

                # Combine into one DataFrame
                df = pd.concat([subjects, nodal, ados], axis=1)

                X = df[[nodeName]].values

                # Features and target
                #X = df.drop(columns=["SubjectID", "ADOS"]).values
                y = df["ADOS"].values
                
                rSquared = trainModel(X,y)
                if (rSquared > .05):
                    plotCorrelation(X.flatten(), y, measure, node)
                    meaningfulResults.append([node, measure, f'{rSquared:.4f}'])
    print(meaningfulResults) 

'''
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Lasso regression
    lasso = Lasso(alpha=0.1, max_iter=10000)
    lasso.fit(X_train, y_train)

    print("=== Lasso Regression ===")
    print("Intercept:", lasso.intercept_)
    print("Test R^2:", lasso.score(X_test, y_test))

    # Show non-zero coefficients with node labels
    coef = lasso.coef_
    nonzero_idx = np.where(coef != 0)[0]
    print("Number of selected nodes:", len(nonzero_idx))
    for i in nonzero_idx:
        print(f"{nodal.columns[i]}: {coef[i]:.4f}")

    # LassoCV for automatic alpha selection
    lasso_cv = LassoCV(alphas=[0.001, 0.01, 0.1, 1, 10], cv=5, max_iter=10000)
    lasso_cv.fit(X_scaled, y)

    print("\n=== LassoCV ===")
    print("Best alpha:", lasso_cv.alpha_)
    coef_cv = lasso_cv.coef_
    nonzero_idx_cv = np.where(coef_cv != 0)[0]
    print("Number of selected nodes:", len(nonzero_idx_cv))
    for i in nonzero_idx_cv:
        print(f"{nodal.columns[i]}: {coef_cv[i]:.4f}")
'''