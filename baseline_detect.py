import pandas as pd
import numpy as np 

df = pd.read_csv("anomalous_reactor_data.csv")
mean_temp = df['Temperature'].mean()
std_temp = df['Temperature'].std()
df['Z_Score'] = (df['Temperature'] - mean_temp) / std_temp
baseline_anomalies = df[df['Z_Score'].abs() > 3]

print(baseline_anomalies.head())