import helper
import os
import sys
import numpy as np
import pandas as pd

for file in os.listdir(sys.argv[1]):
    path = os.getcwd() + '/' + sys.argv[1] + '/' + file
    arr = np.loadtxt(path)
    corr = np.corrcoef(arr.T)
    df = pd.DataFrame(corr)

    new_order = list(df.index[::2]) + list(df.index[1::2])
    df = df.loc[new_order, new_order]

    df.to_csv(path+".csv", header=False, index=False)

    heatmap = helper.csv_to_np(path+".csv")
    helper.fc_to_heatmap(heatmap, path)
