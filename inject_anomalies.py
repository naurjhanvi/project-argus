import pandas as pd
import numpy as np
df = pd.read_csv("normal_reactor_data.csv")
df['label'] = 0
df.loc[500, 'Temperature'] = 500.0
df.loc[500, 'label'] = 1
df.loc[800:850, 'Temperature'] = 300.0
df.loc[800:850, 'label'] = 1
#df['Temp_Variance'] = df['Temperature'].rolling(window=5).var().fillna(0)
df.to_csv("anomalous_reactor_data.csv", index = False)

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(df['Temperature'], label='Temperature', color='blue')
plt.plot(df['Pressure'], label='Pressure', color='orange', alpha=0.5)
anomalies = df[df['label'] == 1]
plt.scatter(anomalies.index, anomalies['Temperature'], color='red', label='Anomaly')
plt.title("Reactor Sensor Data with Injected Faults")
plt.xlabel("time")
plt.ylabel("sensor value")
plt.legend()
plt.show()