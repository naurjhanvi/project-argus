import numpy as np
import pandas as pd

time = np.arange(1000)
#creating data points from normal distribution
temp = np.random.normal(loc = 300.0, scale = 2.0, size = 1000)
random_noise = np.random.normal(loc = 0.0, scale = 0.5, size = 1000)
pressure = (temp * 0.5) + random_noise

data = {
    "Time": time,
    "Temperature" : temp,
    "Pressure" : pressure
}

df = pd.DataFrame(data)
#df['Temp_Variance'] = df['Temperature'].rolling(window=5).var().fillna(0)
df.to_csv("normal_reactor_data.csv", index=False)

print(df.head())