import argparse
import cv2
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import wandb
from tensorflow.keras.applications import ResNet50, ResNet101V2, Xception, InceptionV3
from tensorflow.keras.preprocessing import image

from wandb.keras import WandbCallback

import sys

# import local helpers
sys.path.append('/workspace/app/src')

from data.dataset_helper import *
import params

# sns.set()
# Set Paths

def build_model(imported_model, use_pretrain, output_activation, metrics, output_bias=None, num_classes=1):
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
    output = tf.keras.layers.Dense(num_classes, activation=output_activation, bias_initializer=output_bias)(clf)

    # define new model
    model = tf.keras.models.Model(inputs=model.inputs, outputs=output)

    if num_classes > 1:
        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                      optimizer='sgd',
                      metrics=metrics)
    else:
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


def run_model(batch_size=32, epochs=50, upweight=False, arch="ResNet50", pretrain=False, augment=False,
              percent=10, evaluate=False, downsample="50/50", activation="sigmoid", num_classes=1):
    if downsample == "50/50":
        # using balanced dataset
        train_size = (params.BALANCED_TRAIN_SIZE // 100) * percent
        val_size = (params.BALANCED_VAL_SIZE // 100) * percent
        # test_size = constants.BALANCED_TEST_SIZE

        training_filenames = os.path.join(params.TFR_PATH, "50-50/irrigation", params.BALANCED_TRAINING_FILENAMES)
        validation_filenames = os.path.join(params.TFR_PATH, "50-50/irrigation", params.BALANCED_VALIDATION_FILENAMES)
        # test_filenames = os.path.join(params.TFR_PATH, "50-50/irrigation", constants.BALANCED_TEST_FILENAMES)
    elif downsample == "10/90":
        # using less balanced dataset
        train_size = (params.DOWNSAMPLED_TRAIN_SIZE // 100) * percent
        val_size = (params.DOWNSAMPLED_VAL_SIZE // 100) * percent
        # test_size = constants.BALANCED_TEST_SIZE

        training_filenames = os.path.join(params.TFR_PATH, "10-90/irrigation", params.DOWNSAMPLED_TRAINING_FILENAMES)
        validation_filenames = os.path.join(params.TFR_PATH, "10-90/irrigation",
                                            params.DOWNSAMPLED_VALIDATION_FILENAMES)
        # test_filenames = os.path.join(params.TFR_PATH, "10-90/irrigation", constants.DOWNSAMPLED_TEST_FILENAMES)
    else:
        upweight = False
        # using imbalanced dataset
        train_size = (params.IMBALANCED_TRAIN_SIZE // 100) * percent
        val_size = (params.IMBALANCED_VAL__SIZE // 100) * percent

        training_filenames = os.path.join(params.TFR_PATH, "original", params.IMBALANCED_TRAINING_FILENAMES)
        validation_filenames = os.path.join(params.TFR_PATH, "original", params.IMBALANCED_VALIDATION_FILENAMES)

    wandb.config.update({'dataset.train': train_size})
    wandb.config.update({'dataset.val': val_size})

    arch_dict = {'ResNet50': ResNet50,
                 'ResNet101V2': ResNet101V2,
                 'Xception': Xception,
                 'InceptionV3': InceptionV3}

    architecture = arch_dict[arch]

    if upweight:
        # approximation
        downsampling_factor = params.IMBALANCED_TRAIN_SIZE // params.BALANCED_TRAIN_SIZE
        # wandb.config.update({'downsample': downsampling_factor})

        # class weight = original weight * downsampling factor
        neg = 38400 - 984
        pos = 984
        total = neg + pos
        neg_weight = (1 / neg) * (total) / 2.0
        pos_weight = (1 / pos) * (total) / 2.0
        class_weight = {0: 1,
                        1: downsampling_factor}
        print(f"Using Class Weights (downsample factor: {downsampling_factor}")
        print('\tWeight for Negative Class: {:.2f}'.format(neg_weight))
        print('\tWeight for Positive Class: {:.2f}'.format(pos_weight))

        wandb.config.update({'class_weight': downsampling_factor})

    else:
        class_weight = None

    training_data = get_batched_dataset(training_filenames, batch_size, shuffle=True, num_classes=num_classes)
    val_data = get_batched_dataset(validation_filenames, batch_size=batch_size, shuffle=False, num_classes=num_classes)

    steps_per_epoch = train_size // batch_size
    validation_steps = val_size // batch_size

    # Use an early stopping callback and our timing callback
    if num_classes == 1:
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            verbose=1,
            patience=15,
            mode='max',
            restore_best_weights=True)
    else:
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_precision',
            verbose=1,
            patience=15,
            mode='max',
            restore_best_weights=True)

    time_callback = TimeHistory()

    model = build_model(imported_model=architecture,
                        use_pretrain=pretrain,
                        output_activation=activation,
                        num_classes=num_classes,
                        metrics=params.METRICS
                        )

    if augment:
        # [todo] not working
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
        model.fit(training_data.repeat(),
                  epochs=epochs,
                  steps_per_epoch=steps_per_epoch,
                  validation_data=val_data.repeat(),
                  validation_steps=validation_steps,
                  callbacks=[time_callback, early_stop, WandbCallback()],
                  class_weight=class_weight)

    # Save model to wandb
    model.save(os.path.join(wandb.run.dir, "model.h5"))

    if evaluate:
        test_data = get_batched_dataset(params.TEST_FILENAMES, batch_size=batch_size, shuffle=False, num_classes=num_classes)
        test_steps = params.TEST_SIZE // batch_size

        # callback on evaluation seems to override validation results (maybe that is good things)
        # perf = model.evaluate(test_data, batch_size = batch_size, steps=test_steps, callbacks=[WandbCallback()])
        perf = model.evaluate(test_data, batch_size=batch_size, steps=test_steps, return_dict=False)
        print(perf)

        wandb.config.update({'dataset.test': params.TEST_SIZE})

        wandb.run.summary["test_loss"] = perf[0]
        wandb.run.summary["test_tp"] = perf[1]
        wandb.run.summary["test_fp"] = perf[2]
        wandb.run.summary["test_tn"] = perf[3]
        wandb.run.summary["test_fn"] = perf[4]
        wandb.run.summary["test_accuracy"] = perf[5]
        wandb.run.summary["test_precision"] = perf[6]
        wandb.run.summary["test_recall"] = perf[7]
        wandb.run.summary["test_auc"] = perf[8]
        # wandb.run.summary["test_tfa_f1"] = perf[9]
        # wandb.run.summary["test_tfa_f05"] = perf[10]
        # wandb.run.summary["test_tfa_f2"] = perf[10]
        # wandb.run.summary["test_tfa_f6"] = perf[12]

        if (perf[6] + perf[7]) == 0:
            wandb.run.summary["test_f1"] = 0
        else:
            wandb.run.summary["test_f1"] = 2 * perf[6] * perf[7] / (perf[6] + perf[7])

    return


if __name__ == '__main__':

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


    parser = argparse.ArgumentParser(description='Script for running different supervised classifiers')
    parser.add_argument('-a', '--architecture', choices=['ResNet50', 'ResNet101V2', 'Xception', 'InceptionV3'],
                        help='Class of Model Architecture to use for classification')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        help="batch size to use during training and validation")
    parser.add_argument('-e', '--epochs', default=50, type=int,
                        help="number of epochs to run")
    parser.add_argument('-u', '--upweight', default=False, type=str2bool,
                        help="whether to use weights")
    # parser.add_argument('-g', '--augment', default=False, type=bool,
    #                     help="whether to augment the training data")
    parser.add_argument('-p', '--percent', default=10, type=int,
                        help="portion of datasets to be used for training. 1~100")
    parser.add_argument('-t', '--test', default=False, type=str2bool,
                        help="evaluate the model with test dataset")
    # parser.add_argument('--pretrain', default=False, type=bool,
    #                     help="use imagenet pretrained model")
    parser.add_argument('-d', '--downsample', default="50/50", type=str,
                        help="50/50, 10/90, no")
    parser.add_argument('-c', '--classes', default="1", type=int,
                        help="bnumber of classes. 1 or 43")
    parser.add_argument('-o', '--output_activation', default='sigmoid', choices=['sigmoid', 'softmax', 'relu', 'tanh'],
                        help='output layer of activation func to use for classification')

    args = parser.parse_args()

    if args.classes == 1:
        wandb.init(project="irrigation_detection", entity="cal-capstone")
    else:
        wandb.init(project="BigEarthNet_Classification", entity="cal-capstone")

    wandb.config.update(args)  # adds all of the arguments as config variables
    wandb.config.update({'framework': f'TensorFlow {tf.__version__}'})

    run_model(batch_size=args.batch_size,
              epochs=args.epochs,
              upweight=args.upweight,
              arch=args.architecture,
              pretrain=False,
              augment=False,
              percent=args.percent,
              evaluate=args.test,
              downsample=args.downsample,
              activation=args.output_activation,
              num_classes=args.classes
              )
