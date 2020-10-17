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
from tensorflow.keras.applications import ResNet50, ResNet101V2, Xception, InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import *
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from utils import *
import helpers
import losses
import argparse
import cv2

print(f'Using TensorFlow Version: {tf.__version__}')

# Set Paths
BASE_PATH = './BigEarthData'
OUTPUT_PATH = os.path.join(BASE_PATH, 'models')
TFR_PATH = os.path.join(BASE_PATH, 'tfrecords')

def get_training_dataset(training_filenames, batch_size):
  return get_batched_dataset(training_filenames, batch_size)


def build_simclr_model(imported_model, hidden_1, hidden_2, hidden_3):
  
  base_model = imported_model(include_top=False, weights=None, input_shape=[120,120, 10])
  base_model.trainable = True
  
  inputs = Input((120,120, 10))
  
  h = base_model(inputs, training=True)
  h = GlobalAveragePooling2D()(h)
  
  projection_1 = Dense(hidden_1)(h)
  projection_1 = Activation("relu")(projection_1)
  projection_2 = Dense(hidden_2)(projection_1)
  projection_2 = Activation("relu")(projection_2)
  projection_3 = Dense(hidden_3)(projection_2)

  simclr_model = tf.keras.models.Model(inputs, projection_3)
  
  return simclr_model
          
@tf.function
def train_step(xis, xjs, model, optimizer, criterion, temperature, batch_size):
    # Mask to remove positive examples from the batch of negative samples
    negative_mask = helpers.get_negative_mask(batch_size)
  
    with tf.GradientTape() as tape:
        zis = model(xis)
        zjs = model(xjs)

        # normalize projection feature vectors
        zis = tf.math.l2_normalize(zis, axis=1)
        zjs = tf.math.l2_normalize(zjs, axis=1)

        l_pos = losses._dot_simililarity_dim1(zis, zjs)
        l_pos = tf.reshape(l_pos, (batch_size, 1))
        l_pos /= temperature

        negatives = tf.concat([zjs, zis], axis=0)

        loss = 0

        for positives in [zis, zjs]:
            l_neg = losses._dot_simililarity_dim2(positives, negatives)

            labels = tf.zeros(batch_size, dtype=tf.int32)

            l_neg = tf.boolean_mask(l_neg, negative_mask)
            l_neg = tf.reshape(l_neg, (batch_size, -1))
            l_neg /= temperature

            logits = tf.concat([l_pos, l_neg], axis=1) 
            loss += criterion(y_pred=logits, y_true=labels)

        loss = loss / (2 * batch_size)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss    

def run_model(name, BATCH_SIZE=32, epochs=50, architecture=InceptionV3):
    
    print(50 * "*")
    print(f"Running model: SimCLR {name}")
    print(50 * "=")
    print(f"Batch Size: {BATCH_SIZE}")
    print(50 * "=")
    print(f'Using Model Architecture: {architecture}')
    
    training_filenames = f'{TFR_PATH}/balanced_train_0.tfrecord'
    training_data = (training_filenames, BATCH_SIZE)

#     len_train_records = 9942*5
#     steps_per_epoch = len_train_records // BATCH_SIZE
    
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, 
                                                          reduction=tf.keras.losses.Reduction.SUM)
    decay_steps = 1000
    lr_decayed_fn = tf.keras.experimental.CosineDecay(
        initial_learning_rate=0.1, decay_steps=decay_steps)
    optimizer = tf.keras.optimizers.SGD(lr_decayed_fn)

    simclr_2 = build_simclr_model(architecture,1024, 512, 128)
    simclr_2.summary()

    step_wise_loss = []
    epoch_wise_loss = []
    
    time_callback = TimeHistory()
    augment = Augment()
    
    datagen = image.ImageDataGenerator(
            rotation_range=180,
            width_shift_range=0.10,
            height_shift_range=0.10,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.20,
            preprocessing_function= augment.augfunc)
    
    for epoch in tqdm(range(epochs)):
        for image_batch in tqdm(training_data):
            a = datagen.flow(image_batch, batch_size=BATCH_SIZE, shuffle=False)
            b = datagen.flow(image_batch, batch_size=BATCH_SIZE, shuffle=False)

            loss = train_step(a[0][0], b[0][0], simclr_2, optimizer, criterion, temperature=0.1, batch_size=BATCH_SIZE)
            step_wise_loss.append(loss)

        epoch_wise_loss.append(np.mean(step_wise_loss))
        wandb.log({"nt_xentloss": np.mean(step_wise_loss)})
        
        if epoch % 10 == 0:
            print("epoch: {} loss: {:.3f}".format(epoch + 1, np.mean(step_wise_loss)))
    
    
    epoch_wise_loss, simclr  = train_simclr(simclr_2,
                                            train_ds,
                                            optimizer,
                                            criterion,
                                            temperature=0.1,
                                            epochs=epochs)
    
    
    simclr.save(f'{OUTPUT_PATH}/{name}.h5')
    df = pd.DataFrame(epoch_wise_loss)
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
    args = parser.parse_args()

    arch_dict = {'ResNet50': ResNet50,
                 'ResNet101V2':ResNet101V2,
                 'Xception':Xception,
                 'InceptionV3':InceptionV3}
        
    run_model(args.output,
                  BATCH_SIZE=args.BATCH_SIZE,
                  epochs=args.EPOCHS,
                  architecture=arch_dict[args.arch])

