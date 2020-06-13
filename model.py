import tensorflow as tf 
import numpy as np 

def simple_model():

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation ='relu'),
        tf.keras.layers.Dense(10, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.SGD(lr = 1e-8, momentum = 0.9)
    model.compile(
        loss = tf.keras.losses.Huber(),
        optimizer = optimizer
    )
    return model

