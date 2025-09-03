import numpy as np
import pandas as pd
import random
import csv

import os
import sys

if (len(sys.argv) != 2): 
    sys.exit()

# 1. Load matrix from CSV (assuming no headers in CSV)
connectomes = sys.argv[1]  # shape (116, 116)

r = c = 58

count = 0
avgASD = 0
avgControl = 0

offsets = [0]#[0, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,-1,-2,-3,-4,-5,-6,-7.-8,-9,-10]

# Calculate variance for each subject
for ofs in offsets:
    subjects = []
    variance = []
    count = 0
    avgASD = 0
    avgControl = 0
    for file in os.listdir(connectomes):
        subject = os.path.basename(file).split('_')[1]
        subjects.append(subject)
        A = pd.read_csv(connectomes + file).values
        A12 = A[:r, c:]
        A21 = A[r:, :c]
            
        # Extract diagonals
        #d12 = A12.diagonal(offset = ofs)
        #d21 = A21.diagonal(offset = ofs)
        d12 = np.diag(A12)
        d21 = np.diag(A21)
        all_diags = np.concatenate([d12, d21])

        var_d12 = np.var(all_diags, ddof=1)
        variance.append(var_d12)
        count = count + 1
        if count < 74:
            print(subject, "ASD:", var_d12)
            avgASD += var_d12
        else:
            print(subject, "control:", var_d12)
            avgControl += var_d12
        #print(subject, var_d12)

    filename = "NYU_variance_test3.csv"

    avgASD /= 73
    avgControl /= 98
    print("asd: ", avgASD)
    print("control", avgControl, "\n")

with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["subject", "variance", avgASD, avgControl])   # header row (optional)
    for a, b in zip(subjects, variance):
        writer.writerow([a, b])

