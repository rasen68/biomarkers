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

# Calculate variance for each subject
subjects = []
variance = []

for file in os.listdir(connectomes):
    subject = os.path.basename(file).split('_')[1]
    subjects.append(subject)
    A = pd.read_csv(connectomes + file, header=None).values
    A12 = A[:r, c:]
    A21 = A[r:, :c]
        
    # Extract diagonals
    d12 = np.triu(A12)
    #d12 = A12.diagonal(offset = ofs)
    #d21 = A21.diagonal(offset = ofs)
    #d12 = np.diag(A12)
    #d21 = np.diag(A21)
    #all_diags = np.concatenate([d12, d21])

    #var_d12 = np.var(all_diags, ddof=1)
    var_d12 = np.var(d12, ddof=1)
    variance.append(var_d12)

filename = "NYU_variance_test_abc.csv"

with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["subject", "variance","=AVERAGE(B2, B74)","=AVERAGE(B75, B172)"])   # header row (optional)
    for a, b in zip(subjects, variance):
        writer.writerow([a, b])

