import tensorflow as tf 
import numpy as np 



from data_reader import load_data


series, time = load_data()

split_time = 2000
time_train = time[:split_time]
X_train = series[:split_time]
time_valid = time[split_time:]
X_valid = series[split_time:]

window_size = 30
batch_size = 32
shuffle_buffer_size  = 1000
