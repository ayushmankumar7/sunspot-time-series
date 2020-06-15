import tensorflow as tf 
import numpy as np 

def simple_model(window_size):

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(100, activation ='relu', input_shape = [window_size]),
        tf.keras.layers.Dense(10, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.SGD(lr = 1e-8, momentum = 0.9)
    model.compile(
        loss = tf.keras.losses.Huber(),
        optimizer = optimizer,
        # loss = 'mse'
    )
    print("MODEL CALLED !")
    return model

def lstm_model(window_size):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = -1), input_shape = [window_size]),
        tf.keras.layers.LSTM(32,  return_sequences = True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.SGD(lr = 1e-8, momentum = 0.9)
    model.compile(
        loss = tf.keras.losses.Huber(),
        optimizer = optimizer,
        # loss = 'mse'
    )

    return model

def conv_lstm(window_size):


    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=60, kernel_size=5,
                        strides=1, padding="causal",
                        activation="relu",
                        input_shape=[None, 1]),
    tf.keras.layers.LSTM(60, return_sequences=True),
    tf.keras.layers.LSTM(60, return_sequences=True),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 400)
    ])


    optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)
    model.compile(loss=tf.keras.losses.Huber(),
                optimizer=optimizer,
                metrics=["mae"])

    return model