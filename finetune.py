import pandas as pd
import tensorflow as tf
from glob import glob
import os
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import csv
import json
import time
from tensorflow.keras.layers import *
from utils import *
import helpers
import losses
import argparse
import cv2
from pprint import pprint

# Set Paths
BASE_PATH = './BigEarthData'
OUTPUT_PATH = os.path.join(BASE_PATH, 'models')
TFR_PATH = os.path.join(BASE_PATH, 'tfrecords')

METRICS = [
          tf.keras.metrics.TruePositives(name='tp'),
          tf.keras.metrics.FalsePositives(name='fp'),
          tf.keras.metrics.TrueNegatives(name='tn'),
          tf.keras.metrics.FalseNegatives(name='fn'),
          tf.keras.metrics.BinaryAccuracy(name='accuracy'),
          tf.keras.metrics.Precision(name='precision'),
          tf.keras.metrics.Recall(name='recall'),
          tf.keras.metrics.AUC(name='auc'),
      ] 

def get_training_dataset(training_filenames, batch_size):
  return get_batched_dataset(training_filenames, batch_size)

def get_validation_dataset(validation_filenames, batch_size):
  return get_batched_dataset(validation_filenames, batch_size)

def load_pretrained_model(model, metrics=METRICS, hidden1=256, hidden2=256):
  
  pretrained_model = tf.keras.models.load_model(model)
  pretrained_model.trainable = False  

  h1 = tf.keras.layers.Dense(hidden1, activation='elu', name='dense_ft_1')(pretrained_model.layers[-2].output)
  h1 = tf.keras.layers.Dropout(0.50)(h1)
  h2 = tf.keras.layers.Dense(hidden2, activation='elu', name='dense_ft_2')(h1)
  h2 = tf.keras.layers.Dropout(0.50)(h2)
  output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(h2)
  
  # define new model
  new_model = tf.keras.models.Model(inputs=pretrained_model.inputs, outputs=output)

  ft_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)  

  new_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=ft_optimizer,
              metrics=metrics)


  return new_model

def run_model(name, pretrained_model, BATCH_SIZE, epochs, training_dataset):
    print(50 * "*")
    print(f"Running model: {name}")
    print(50 * "=")
    print(f"Batch Size: {BATCH_SIZE}")

    training_filenames = f'{TFR_PATH}/{training_dataset}'
    validation_filenames = f'{TFR_PATH}/balanced_val.tfrecord'

    training_data = get_training_dataset(training_filenames, batch_size=BATCH_SIZE)
    val_data = get_validation_dataset(validation_filenames, batch_size=BATCH_SIZE)

    len_val_records = 4384 
    if training_dataset == 'balanced_train_13percent.tfrecord':
      len_train_records = 640
    elif training_dataset == 'balanced_train_3percent.tfrecord':
      len_train_records = 128
    elif trainin_dataset == 'balanced_train_1percent.tfrecord':
      len_train_records = 64
    else:
      len_train_records = 9942 * 5
    
    steps_per_epoch = len_train_records // BATCH_SIZE
    validation_steps = len_val_records // BATCH_SIZE
    
    print(f'Using {training_dataset} as training data.')
    print(f'{len_train_records} total records and {steps_per_epoch} steps per epoch')


    # Use an early stopping callback and our timing callback
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        verbose=1,
        patience=25,
        mode='max',
        restore_best_weights=True)

    time_callback = TimeHistory()
    
    print(f'Using Pretrained Model: {pretrained_model}')
          
    model = load_pretrained_model(pretrained_model)
    model.summary()

    history = model.fit(training_data,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=val_data,
                        validation_steps=validation_steps,
                        callbacks=[time_callback, early_stop])
    times = time_callback.times
    df = pd.DataFrame(history.history)
    df['times'] = time_callback.times
    
    df.to_pickle(f'{OUTPUT_PATH}/{name}.pkl')
    model.save(f'{OUTPUT_PATH}/{name}.h5')
      

    return df
  
if __name__ == '__main__':
    
    print('In main function')
    parser = argparse.ArgumentParser(description='Script for running different supervised classifiers')
    parser.add_argument('-p', '--pretrained',
                        choices=[file for file in os.listdir('./BigEarthData/models/') if file[-3:]=='.h5' if file[:6]=='simclr'],
                        help='Which pretrained model do you want to finetune?')
    parser.add_argument('-o', '--output', type=str,
                        help='Output File Prefix for model file and dataframe')
    parser.add_argument('-b', '--BATCH_SIZE', default=32, type=int,
                       help="batch size to use during training and validation")
    parser.add_argument('-e', '--EPOCHS', default=10, type=int,
                        help="number of epochs to run")
    parser.add_argument('-t', '--training_data', type=str,
                        choices=['balanced_train_13percent.tfrecord',
                                 'balanced_train_3percent.tfrecord',
                                 'balanced_train_1percent.tfrecord',
                                 'balanced_train_*'])
    
    args = parser.parse_args()
    
    
    print(f'Using TensorFlow Version: {tf.__version__}')


    
    model_path = os.path.join(OUTPUT_PATH, args.pretrained)

    run_model(name=args.output,
              pretrained_model=model_path,
              BATCH_SIZE=args.BATCH_SIZE,
              epochs=args.EPOCHS,
              training_dataset=args.training_data)
    
    
