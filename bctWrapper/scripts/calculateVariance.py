import numpy as np
import pandas as pd
import random
import csv
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns



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

group1 = variance[73:]
group2 = variance[:73]
t_stat, p_value = ttest_ind(group1, group2, equal_var = False)
print(t_stat)
print(p_value)

# Plot results
plt.figure(figsize=(6,4))
sns.violinplot(data=[group1, group2])
sns.swarmplot(data=[group1, group2], color=".25")
plt.xticks([0,1], ["ASD", "Control"])
plt.ylabel("Variance")
plt.show()

'''
with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["subject", "variance","=AVERAGE(B2, B74)","=AVERAGE(B75, B172)"])   # header row (optional)
    for a, b in zip(subjects, variance):
        writer.writerow([a, b])

'''