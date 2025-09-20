from scipy.stats import ttest_ind
import pandas as pd
import numpy as np
import seaborn as sns
import sys
import csv
import os

featuresList = ['modularity'] #pd.read_csv(sys.argv[1]).columns.tolist()[1:]
graphTheoryMeasures = pd.read_csv(sys.argv[1])

tVals = []
pVals = []

for feature in featuresList:
    asdGTMeasure = pd.DataFrame(graphTheoryMeasures)[[feature]][:73].values.flatten().tolist()
    controlGTMeasure = pd.DataFrame(graphTheoryMeasures)[[feature]][73:].values.flatten().tolist()

    print("Feature:", feature)
    #print(asdGTMeasure)
    #print(controlGTMeasure)
    t_stat, p_value = ttest_ind(asdGTMeasure, controlGTMeasure)
    print("t value:", t_stat)
    print("p value:", p_value, "\n")
    tVals.append(t_stat)
    pVals.append(p_value)

with open("ttest_new.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["feature", "ttest", "pvalue"])   # header row (optional)
    for a, b, c in zip(featuresList, tVals, pVals):
        writer.writerow([a, b, c])

