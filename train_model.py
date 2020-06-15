import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

from model import simple_model, lstm_model, conv_lstm
from utils import windowed_dataset, plot_series
from data_reader import load_data


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

print(X_train.shape)
dataset = windowed_dataset(X_train, window_size, batch_size, shuffle_buffer_size)
print(dataset)

#  IMPORT MODEL
model = conv_lstm(window_size)

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10 ** (epoch/20))

print("THIS REACHED ")
history = model.fit(dataset, epochs = 200, verbose = 0, callbacks = [lr_schedule])
print(history)
model.save('saved_models/conv_lstm.h5')

