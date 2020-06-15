import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
from utils import plot_series, model_forecast
from data_reader import load_data

model = tf.keras.models.load_model('saved_models/conv_lstm.h5', custom_objects={'tf': tf})

print("\n M O D E L   S U M M A R Y  \n")
print(model.summary())




series, time = load_data()

split_time = 2000
time_train = time[:split_time]
X_train = series[:split_time]
time_valid = time[split_time:]
X_valid = series[split_time:]

window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

# plot_series(time, series, title = "Original Data")
# plt.show()




print("\n  Please be patient! _()_  This might take some time. \n")

# forecast = []
# for time in range(len(series) - window_size):
#   forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

# forecast = forecast[split_time-window_size:]
# results = np.array(forecast)[:, 0, 0]

rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]

plt.figure(figsize=(10, 6))


plot_series(time_valid, X_valid)
plot_series(time_valid, rnn_forecast, title = "conv_lstm prediction", text = "Conv1D(32)\nLSTM(32)\nLSTM(32)\nDense(1)\nloss = Huber\nOptimizer=SGD")
plt.show()

# plt.savefig('plotted_graphs/simple_model.png', bbox_inches='tight')