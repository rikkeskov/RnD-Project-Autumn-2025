import pandas as pd
import numpy as np

file1 = "BIDMC32_TEST_dimAVR_e0.0001_eb0.03125_decompressed.csv"
file2 = "BIDMC32_TEST_dimAVR_e0.03_eb0.03125_decompressed.csv"

df1 = pd.read_csv("data/decompressed/" + file1, header=None)
df2 = pd.read_csv("data/decompressed/" + file2, header=None)

print("Same shape:")
print(df1.shape, df2.shape)

print("Exactly equal:")
print(df1.equals(df2))

print("Maximum difference:")
print(np.max(np.abs(df1.values - df2.values)))

print("Mean difference:")
print(np.mean(np.abs(df1.values - df2.values)))