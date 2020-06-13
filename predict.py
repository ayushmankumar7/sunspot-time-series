import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
from utils import plot_series
from data_reader import load_data

model = tf.keras.models.load_model('saved_models/simple_model.h5')

print("\n M O D E L   S U M M A R Y  \n")
print(model.summary())




series, time = load_data()

split_time = 3000
time_train = time[:split_time]
X_train = series[:split_time]
time_valid = time[split_time:]
X_valid = series[split_time:]

plot_series(time, series)
plt.show()

window_size = 30
batch_size = 32
shuffle_buffer_size  = 1000


print("\n  Please be patient! _()_  This might take some time. \n")

forecast = []
for time in range(len(series) - window_size):
  forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:, 0, 0]


plt.figure(figsize=(10, 6))


plot_series(time_valid, X_valid)

plot_series(time_valid, results)
plt.show()