import argparse
# import seaborn as sns
# from matplotlib.cm import get_cmap
import csv
import cv2
import json
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import time
import wandb
from glob import glob
from matplotlib import pyplot as plt
from tensorflow.keras.applications import ResNet50, ResNet101V2, Xception, InceptionV3
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
from wandb.keras import WandbCallback

from utils import *

print(f'Using TensorFlow Version: {tf.__version__}')
# sns.set()

# Set Paths
BASE_PATH = '/workspace/app'
OUTPUT_PATH = os.path.join(BASE_PATH, 'models/supervised')
TFR_PATH = os.path.join(BASE_PATH, 'data/processed')


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

def build_model(imported_model, use_pretrain, metrics=METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    if use_pretrain:
        # This option cannot actually be used due to incompatibility with input tensor shapes
        model = imported_model(include_top=False, weights='imagenet', input_tensor=None, input_shape=[120, 120, 10],
                               pooling=None)
        model.trainable = False
    else:
        model = imported_model(include_top=False, weights=None, input_tensor=None, input_shape=[120, 120, 10],
                               pooling=None)
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
    #   print(f'Trainable variables: {model.trainable_weights}')

    return model


class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def augfunc(sample):
    # Randomly apply transformation (color distortions) with probability p.
    sample = _random_apply(_color_jitter, sample, p=0.8)
    sample = _random_apply(_color_drop, sample, p=0.2)
    sample = _random_apply(_blur, sample, p=0.5)

    return sample


def _color_jitter(x, s=1):
    # one can also shuffle the order of following augmentations
    # each time they are applied.
    x = tf.image.random_brightness(x, max_delta=0.8 * s)
    x = tf.image.random_contrast(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    dx = tf.image.random_saturation(x[:, :, :3], lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    dx = tf.image.random_hue(dx, max_delta=0.2 * s)
    x = tf.concat([dx, x[:, :, 3:]], axis=2)
    x = tf.clip_by_value(x, 0, 1)
    return x


def _color_drop(x):
    dx = tf.image.rgb_to_grayscale(x[:, :, :3])
    dx = tf.tile(dx, [1, 1, 3])
    x = tf.concat([dx, x[:, :, 3:]], axis=2)
    return x


def _blur(x):
    # SimClr implementation is applied at 10% of image size with a random sigma
    p = np.random.uniform(0.1, 2)
    if type(x) == np.ndarray:
        return (cv2.GaussianBlur(x, (5, 5), p))
    return (cv2.GaussianBlur(x.numpy(), (5, 5), p))


def _random_apply(func, x, p):
    return tf.cond(
        tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
                tf.cast(p, tf.float32)),
        lambda: func(x),
        lambda: x)


def run_model(prefix, batch_size=32, epochs=50, weights=False, architecture="ResNet50", pretrain=False, augment=False, percent=10, evaluate=False):

    # previous team code was running on a lot smaller set
    # len_train_records = 4384*2
    # len_val_records = 4384
    # len_test_records = 4384
    len_train_records = (269695 // 100) * percent
    len_val_records = (123723 // 100) * percent
    len_test_records = 125866

    # 1. Start a W&B run
    # name = f"BE supervised {architecture} b{batch_size} e{epochs}"
    # wandb.init(project="irrigation_detection", name=name)
    wandb.init(project="irrigation_detection")
    wandb.config.epochs = epochs
    wandb.config.batch_size = batch_size
    wandb.config.architecture = architecture
    wandb.config.update({'dataset.percent': percent})
    wandb.config.update({'dataset.train': len_train_records})
    wandb.config.update({'dataset.val': len_val_records})

    arch_dict = {'ResNet50': ResNet50,
                 'ResNet101V2': ResNet101V2,
                 'Xception': Xception,
                 'InceptionV3': InceptionV3}

    architecture = arch_dict[args.arch]

    # print(50 * "*")
    # print(f"Running model: {name}")
    # print(50 * "=")
    # print(f"Batch Size: {batch_size}")
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

    # training_filenames = f'{TFR_PATH}/balanced_train_3percent.tfrecord'
    # validation_filenames = f'{TFR_PATH}/balanced_val.tfrecord'
    training_filenames = f'{TFR_PATH}/train-*.tfrecord'
    validation_filenames = f'{TFR_PATH}/val-*.tfrecord'
    test_filenames = f'{TFR_PATH}/test-*.tfrecord'

    training_data = get_training_dataset(training_filenames, batch_size=batch_size)
    val_data = get_validation_dataset(validation_filenames, batch_size=batch_size)

    # counting on the fly takes hours if not days
    # len_val_records = val_data.reduce(np.int64(0), lambda x, _: x + 1)
    # print(f"val set: {len_val_records}")

    # len_train_records = training_data.reduce(np.int64(0), lambda x, _: x + 1)
    # print(f"train set: {len_train_records}")

    # len_test_records = test_data.reduce(np.int64(0), lambda x, _: x + 1)
    # print(f"test set: {len_test_records}")

    steps_per_epoch = len_train_records // batch_size
    validation_steps = len_val_records // batch_size

    # Use an early stopping callback and our timing callback
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        verbose=1,
        patience=15,
        mode='max',
        restore_best_weights=True)

    time_callback = TimeHistory()

    # print(f'Using Model Architecture: {architecture}')

    model = build_model(imported_model=architecture,
                        use_pretrain=pretrain)
    # print(f'Trainable variables: {model.trainable_weights}')
    # model.summary()

    if augment:
        print(50 * "*")
        print(f"augmenting")
        print(50 * "=")
        train_df = pd.read_pickle(training_filenames)
        train_X = train_df.X.values
        train_y = train_df.y.values

        train_X = np.stack(train_X)
        train_y = np.stack(train_y)

        datagen = image.ImageDataGenerator(
            rotation_range=180,
            width_shift_range=0.10,
            height_shift_range=0.10,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.20,
            preprocessing_function=augfunc)
        aug_data = datagen.flow(train_X, train_y, batch_size=batch_size, shuffle=True)

        history = model.fit(aug_data,
                            epochs=epochs,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=val_data,
                            validation_steps=validation_steps,
                            callbacks=[time_callback, early_stop, WandbCallback()],
                            class_weight=class_weight)

        # times = time_callback.times
        # df = pd.DataFrame(history.history)
        # df['times'] = time_callback.times

    else:
        model.fit(training_data,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_data,
            validation_steps=validation_steps,
            # callbacks=[WandbCallback()],
            callbacks=[time_callback, early_stop, WandbCallback()],
            class_weight=class_weight)
        # times = time_callback.times
        # df = pd.DataFrame(history.history)
        # df['times'] = time_callback.times

    # df.to_pickle(f'{OUTPUT_PATH}/{name}.pkl')
    # model.save(f'{OUTPUT_PATH}/{name}.h5')

    if evaluate:
        wandb.config.update({'dataset.test': f'{len_test_records}'})

        test_data = get_validation_dataset(test_filenames, batch_size=batch_size)
        test_steps = len_test_records // batch_size

        # [0.01763233356177807, 0.0, 0.0, 4384.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        # perf = model.evaluate(test_data, steps=test_steps, callbacks=[WandbCallback()])
        perf = model.evaluate(test_data, batch_size = batch_size, steps=test_steps)
        wandb.run.summary["test_loss"] = perf[0]
        wandb.run.summary["test_tp"] = perf[1]
        wandb.run.summary["test_fp"] = perf[2]
        wandb.run.summary["test_tn"] = perf[3]
        wandb.run.summary["test_fn"] = perf[4]
        wandb.run.summary["test_accuracy"] = perf[5]
        wandb.run.summary["test_precision"] = perf[6]
        wandb.run.summary["test_recall"] = perf[7]
        wandb.run.summary["test_auc"] = perf[8]
        if (perf[6] + perf[7]) == 0:
            wandb.run.summary["test_f1"] = 0
        else:
            wandb.run.summary["test_f1"] = 2*perf[6]*perf[7]/(perf[6] + perf[7])

    # Save model to wandb
    # model.save(os.path.join(wandb.run.dir, "model.h5"))

    return
    # return df

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script for running different supervised classifiers')
    parser.add_argument('-a', '--arch', choices=['ResNet50', 'ResNet101V2', 'Xception', 'InceptionV3'],
                        help='Class of Model Architecture to use for classification')
    parser.add_argument('-o', '--output', type=str,
                        help='Output File Prefix for model file and dataframe')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        help="batch size to use during training and validation")
    parser.add_argument('-e', '--epochs', default=50, type=int,
                        help="number of epochs to run")
    parser.add_argument('-w', '--weights', default=False, type=bool,
                        help="whether to use weights")
    parser.add_argument('-g', '--augment', default="False", type=str, choices=['True', 'False'],
                        help="whether to augment the training data")
    parser.add_argument('-p', '--percent', default=10, type=int,
                        help="portion of datasets to be used for training. 1~100")
    parser.add_argument('-t', '--test', default="False", type=str, choices=['True', 'False'],
                        help="evaluate the model with test dataset")
    args = parser.parse_args()

    # augment = False
    # if args.augment == 'True':
    #     AUGMENT = True

    run_model(args.output,
              batch_size=args.batch_size,
              epochs=args.epochs,
              weights=False,
              architecture=args.arch,
              pretrain=False,
              augment=False,
              percent=args.percent,
              evaluate=args.test)

