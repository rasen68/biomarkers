import numpy as np
import pandas as pd
import csv

import os
import sys

if (len(sys.argv) != 2): 
    sys.exit()

# 1. Load matrix from CSV (assuming no headers in CSV)
connectomes = sys.argv[1]  # shape (116, 116)

r = c = 58
subjects = []
variance = []

# Calculate variance for each subject
for file in os.listdir(connectomes):
    subject = os.path.basename(file).split('_')[1]
    subjects.append(subject)
    A = pd.read_csv(connectomes + file).values
    A12 = A[:r, c:]
    A21 = A[r:, :c]

    # Extract diagonals
    d12 = np.diag(A12)
    d21 = np.diag(A21)
    all_diags = np.concatenate([d12, d21])

    var_d12 = np.var(all_diags, ddof=1)
    variance.append(var_d12)
    print(subject, var_d12)


with open("variancetester.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["subject", "variance"])   # header row (optional)
    for a, b in zip(subjects, variance):
        writer.writerow([a, b])

