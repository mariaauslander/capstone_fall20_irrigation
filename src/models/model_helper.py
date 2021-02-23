import tensorflow as tf
from tensorflow.keras.layers import *


def load_pretrained_model(model, hidden1=256, hidden2=256):
  
  pretrained_model = tf.keras.models.load_model(model)
  pretrained_model.trainable = True  

  h1 = tf.keras.layers.Dense(hidden1, activation='elu', name='dense_ft_1')(pretrained_model.layers[-2].output)
  h1 = tf.keras.layers.Dropout(0.50)(h1)
  h2 = tf.keras.layers.Dense(hidden2, activation='elu', name='dense_ft_2')(h1)
  h2 = tf.keras.layers.Dropout(0.50)(h2)
  output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(h2)
  
  # define new model
  new_model = tf.keras.models.Model(inputs=pretrained_model.inputs, outputs=output)

  # Learning rate of 5e-5 used for finetuning based on hyperparameter evaluations
  ft_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)  
  
  # Compile model with Cross Entropy loss
  new_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=ft_optimizer,
              metrics=params.METRICS)

  return new_model




