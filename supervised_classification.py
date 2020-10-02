import pandas as pd
import tensorflow as tf
from glob import glob
import os
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
#from matplotlib.cm import get_cmap
import csv
import json
import time
from tensorflow.keras.applications import ResNet50, ResNet101V2, Xception, InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from utils import *
import argparse

print(f'Using TensorFlow Version: {tf.__version__}')
sns.set()

# Set Paths
BASE_PATH = './BigEarthData'
OUTPUT_PATH = os.path.join(BASE_PATH, 'models')
TFR_PATH = os.path.join(BASE_PATH, 'tfrecords')

def get_training_dataset(training_filenames, batch_size):
  return get_batched_dataset(training_filenames, batch_size)

def get_validation_dataset(validation_filenames, batch_size):
  return get_batched_dataset(validation_filenames, batch_size)

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

def build_model(imported_model, use_pretrain, metrics = METRICS, output_bias=None):
  if output_bias is not None:
    output_bias = tf.keras.initializers.Constant(output_bias)
  if use_pretrain:
    model = imported_model(include_top=False, weights='imagenet', input_tensor=None, input_shape=[120,120, 10],  pooling=None)
    model.trainable = False
  else:
    model = imported_model(include_top=False, weights=None, input_tensor=None, input_shape=[120,120, 10],  pooling=None)
    model.trainable = True
  # add new classifier layers
  flat = tf.keras.layers.Flatten()(model.layers[-1].output)
  h1 = tf.keras.layers.Dense(1024, activation='elu')(flat)
  h1 = tf.keras.layers.Dropout(0.25)(h1)
  h2 = tf.keras.layers.Dense(512, activation='elu')(h1)
  h2 = tf.keras.layers.Dropout(0.25)(h2)
  clf = tf.keras.layers.Dense(256, activation='elu')(h2)
  output = tf.keras.layers.Dense(1, activation='sigmoid',
                                 bias_initializer=output_bias)(clf)
  # define new model
  model = tf.keras.models.Model(inputs=model.inputs, outputs=output)

  model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer='adam',
              metrics=metrics)
  return model


class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def run_model(name, BATCH_SIZE=32, epochs=50, weights=False, architecture=ResNet50, pretrain=False):
    print(50 * "*")
    print(f"Running model: {name}")
    print(50 * "=")
    print(f"Batch Size: {BATCH_SIZE}")
    if weights:
        neg = 38400 - 984
        pos = 984
        total = neg + pos
        neg_weight = (1 / neg) * (total) / 2.0
        pos_weight = (1 / pos) * (total) / 2.0
        class_weight = {0: neg_weight,
                        1: pos_weight}
        print(f"Using Class Weights: ")
        print('\tWeight for Negative Class: {:.2f}'.format(neg_weight))
        print('\tWeight for Positive Class: {:.2f}'.format(pos_weight))
    else:
        class_weight = None
        print("Not Using Weights")

    training_filenames = f'{TFR_PATH}/balanced_train_0.tfrecord'
    validation_filenames = f'{TFR_PATH}/val-part-0.tfrecord'

    training_data = get_training_dataset(training_filenames, batch_size=BATCH_SIZE)
    val_data = get_validation_dataset(validation_filenames, batch_size=BATCH_SIZE)

    len_val_records = 4384 
    len_train_records = 9942
    steps_per_epoch = len_train_records // BATCH_SIZE
    validation_steps = len_val_records // BATCH_SIZE

    # Use an early stopping callback and our timing callback
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_precision',
        verbose=1,
        patience=25,
        mode='max',
        restore_best_weights=True)

    time_callback = TimeHistory()

    model = build_model(imported_model=architecture,
                        use_pretrain=pretrain)
    model.summary()

    history = model.fit(training_data,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=val_data,
                        validation_steps=validation_steps,
                        callbacks=[time_callback, early_stop],
                        class_weight=class_weight)
    times = time_callback.times

    model.save(f'{OUTPUT_PATH}/{name}.h5')
    df = pd.DataFrame(history.history)
    df['times'] = time_callback.times
    df.to_pickle(f'{OUTPUT_PATH}/{name}.pkl')

    return df

if __name__ == '__main__':
    
    print('In main function')
    parser = argparse.ArgumentParser(description='Script for running different supervised classifiers')
    parser.add_argument('-a', '--arch', choices=['ResNet50', 'ResNet101V2', 'Xception', 'InceptionV3'],
                        help='Class of Model Architecture to use for classification')
    parser.add_argument('-o', '--output', type=str,
                        help='Output File Prefix for model file and dataframe')
    parser.add_argument('-b', '--BATCH_SIZE', default=32, type=int,
                       help="batch size to use during training and validation")
    parser.add_argument('-e', '--EPOCHS', default=50, type=int,
                        help="number of epochs to run")
    parser.add_argument('-w', '--weights', default=False, type=bool,
                       help="whether to use weights")
    args = parser.parse_args()

    arch_dict = {'ResNet50': ResNet50,
                 'ResNet101V2':ResNet101V2,
                 'Xception':Xception,
                 'InceptionV3':InceptionV3}
    
    run_model(args.output,
                  BATCH_SIZE=args.BATCH_SIZE,
                  epochs=args.EPOCHS,
                  weights=False,
                  architecture=arch_dict[args.arch],
                  pretrain=False)

