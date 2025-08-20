import os
import sys

# Import libraries.
import helpers 
import numpy as np
import pandas as pd
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

from sklearn.datasets import fetch_california_housing

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

if len(sys.argv) != 3:
    sys.exit()

if __name__ == "__main__":
    #housing = fetch_california_housing()

    #XTEST = pd.DataFrame(housing.data, columns=housing.feature_names)[["AveRooms"]]
    #yTEST = housing.target  # Median house value in $100,000s

    #print("Housing X: \n", XTEST)
    #print("Housing Y: \n", yTEST)

    featuresList = pd.read_csv(sys.argv[1]).columns.tolist()[1:]
    graphTheoryMeasures = pd.read_csv(sys.argv[1])

    demographics = ["FIQ", "PIQ", "AGE_AT_SCAN", "SEX", "VIQ", "ADOS_MODULE", "ADOS_TOTAL", "ADOS_COMM", "ADOS_SOCIAL", "ADOS_STEREO_BEHAV"]

    posCorr = []
    posCorr2 = []
    for demo in demographics:
        numSubjects = 170
        if "ADOS" in demo:
            numSubjects = 70
        score = getColumnFromCSV(sys.argv[2], demo)
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
            if r2 > .015:
                posCorr.append(f'GT Measure: {feature}, demographic: {demo}, r2: {r2:.4f}')
                posCorr2.append(f'{r2:.4f}')
                plotCorrelation(X.values.flatten().tolist(), y, feature, demo)
    
    print(posCorr)
    print(posCorr2)


        

