import pandas as pd
import numpy as np
import sys
import os

for file in os.listdir(sys.argv[1]):
    path = sys.argv[1] + file
    df = pd.read_csv(path)
    np.fill_diagonal(df.values, 0)
    df[df < 0] = 0

    df.to_csv(path, index=False)
    print(path, "Done!")
