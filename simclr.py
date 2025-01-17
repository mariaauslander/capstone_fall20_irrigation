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

# Function for getting a dataset generator for our training data. 
# The flags affect which tfrecords files to use and how to normalize each.
def get_training_dataset(training_filenames, batch_size, ca_flag):
  return get_batched_dataset(training_filenames, batch_size, simclr=True, ca=ca_flag)


def build_simclr_model(imported_model, hidden_1, hidden_2, hidden_3):
  '''
  This function is used to actually create the neural encoder and projection head. The
  neural encoder is basically on of ResNet50, ResNet101V2, Xception or InceptionV3 (or any other)
  We train 10 channels of the satellite data. The projection head dimensions should be specified as inputs.
  
  imported_model: tensorflow.keras.applications model - ResNet101V2 is typically used
  hidden_1: integer - dimension of first layer of the projection head
  hidden_2: integer - dimension of second layer of the projection head
  hidden_3: integer - output dimension - vector used in the contrastive loss function
  '''
  
  # Load in a Keras Model for our neural encoder and set to trainable
  base_model = imported_model(include_top=False, weights=None, input_shape=[120,120, 10])
  base_model.trainable = True
  
  # Input dimensions are fixed to big earth net image dimensions.
  inputs = Input((120,120,10))
  
  # Add a Global Average Pooling to flatten the output of the neural encoder
  h = base_model(inputs, training=True)
  h = GlobalAveragePooling2D()(h)
  
  # Add the projection head layers with Relu activations
  projection_1 = Dense(hidden_1)(h)
  projection_1 = Activation("relu")(projection_1)
  projection_2 = Dense(hidden_2)(projection_1)
  projection_2 = Activation("relu")(projection_2)
  projection_3 = Dense(hidden_3)(projection_2)

  # Define our final model and return from function
  simclr_model = tf.keras.models.Model(inputs, projection_3)
  
  return simclr_model
          
@tf.function
def train_step(xis, xjs, model, optimizer, criterion, temperature, batch_size):
    
    # Mask to remove positive examples from the batch of negative samples
    negative_mask = helpers.get_negative_mask(batch_size)
  
    with tf.GradientTape() as tape:
        # Get our latent space vectors for our two sets of augmented images.
        zis = model(xis)
        zjs = model(xjs)

        # normalize projection feature vectors
        zis = tf.math.l2_normalize(zis, axis=1)
        zjs = tf.math.l2_normalize(zjs, axis=1)

        # Similarity between all positive pairs
        l_pos = losses._dot_simililarity_dim1(zis, zjs)
        l_pos = tf.reshape(l_pos, (batch_size, 1))
        
        # Divide by your temperature variable or tau
        l_pos /= temperature
        
        # Combine all images to create negative array 
        negatives = tf.concat([zjs, zis], axis=0)

        loss = 0

        # Compare every image vector to every other image vector 
        for positives in [zis, zjs]:
            
            l_neg = losses._dot_simililarity_dim2(positives, negatives)
            
            # Negative examples have zero label
            labels = tf.zeros(batch_size, dtype=tf.int32)

            # Mask out the positive pairs
            l_neg = tf.boolean_mask(l_neg, negative_mask)
                   
            l_neg = tf.reshape(l_neg, (batch_size, -1))
            l_neg /= temperature

            logits = tf.concat([l_pos, l_neg], axis=1) 
            
            # Cross entropy loss
            loss += criterion(y_pred=logits, y_true=labels)

        loss = loss / (2 * batch_size)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss    

def run_model(name, BATCH_SIZE, epochs, architecture, temperature, ca_flag):
    
    '''
    Main execution function used to take input flags and control overall model flow.
    
    name: -string Output name for model file
    BATCH_SIZE: int- batch size to use during training - set to be large
    epochs: int - number of passes over the data
    architecture: - tensorflow.keras.applications model to use as neural encoder
    temperature: float - temperature for the softmax
    ca_flag: Boolean - specify whether training on California data or BEN data
    '''
    
    # Log information
    print(50 * "*")
    print(f"Running model: SimCLR {name}")
    print(50 * "=")
    print(f"Batch Size: {BATCH_SIZE}")
    print(50 * "=")
    print(f'Using Model Architecture: {architecture}')
    
    # California data has different files
    if ca_flag:
      training_filenames = f'{TFR_PATH}/train_ca_part*.tfrecord'
    else:
      training_filenames = f'{TFR_PATH}/train-part*.tfrecord'
      
    # Get the training files in batches  
    training_data = get_training_dataset(training_filenames, BATCH_SIZE, ca_flag=ca_flag)

    # Use Cross Entropy Loss
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, 
                                                          reduction=tf.keras.losses.Reduction.SUM)
    # Learning Rate Decay with stochastic gradient descent
    decay_steps = 1000
    lr_decayed_fn = tf.keras.experimental.CosineDecay(
        initial_learning_rate=0.1, decay_steps=decay_steps)
    optimizer = tf.keras.optimizers.SGD(lr_decayed_fn)

    # Build the model with the following hidden layer sizes
    simclr_2 = build_simclr_model(architecture,1024, 512, 128)
    
    # Print Summary of model for user
    simclr_2.summary()

    # List for tracking losses by epoch
    epoch_wise_loss = []
    
    # Track time spent per epoch
    time_callback = TimeHistory()
    
    # Augment Class used for color distortion and Gaussian Blur
    augment = Augment()
    
    # Set Other Augmentation data
    ROTATION = 180
    SHIFT = 0.10
    FLIP = True
    ZOOM = 0.20
    JITTER = 0.0
    BLUR = True
    
    # Use Keras to augment images in batches
    datagen = image.ImageDataGenerator(
            rotation_range=ROTATION,
            width_shift_range=SHIFT,
            height_shift_range=SHIFT,
            horizontal_flip=FLIP,
            vertical_flip=FLIP,
            zoom_range=ZOOM,
            preprocessing_function= augment.augfunc)
    
    min_loss = 1e6
    min_loss_epoch = 0
    
    # Manually walk through epochs and batches
    for epoch in tqdm(range(epochs)):
      step_wise_loss = []
      
      # Loop over batches, perform augmentation and calculate poss
      for image_batch in tqdm(training_data):
        # Use the data generator to augment the data - DO NOT SHUFFLE - images need to stay aligned
        a = datagen.flow(image_batch, batch_size=BATCH_SIZE, shuffle=False)
        b = datagen.flow(image_batch, batch_size=BATCH_SIZE, shuffle=False)
        
        # Send image arrays, simclr model, etc to our train_step function
        loss = train_step(a[0][0], b[0][0], simclr_2, optimizer, criterion, temperature=temperature, batch_size=BATCH_SIZE)
        step_wise_loss.append(loss)
      
      # Append to list of loss by epoch
      epoch_wise_loss.append(np.mean(step_wise_loss))

      # Print the loss after every epoch
      print(f"****epoch: {epoch + 1} loss: {epoch_wise_loss[-1]:.3f}****\n")
        
      # Save weights every five epochs
      if (epoch > 0) and ((epoch+1) % 5 == 0):
        print(f'Saving weights for epoch: {epoch+1}')
        # Save the final model with weights
        simclr_2.save(f'{OUTPUT_PATH}/{name}_{epoch+1}.h5')
  
    # Store the epochwise loss and model metadata to dataframe
    df = pd.DataFrame(epoch_wise_loss)
    df['temperature'] = temperature
    df['batch_size'] = BATCH_SIZE
    df['epochs'] = epochs
    df['h1'] = 1024
    df['h2'] = 512
    df['output_dim'] = 128
    df['rotation'] = ROTATION
    df['shift'] = SHIFT
    df['flip'] = FLIP
    df['zoom'] = ZOOM
    df['jitter'] = JITTER
    df['blur'] = BLUR
  
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
    parser.add_argument('-t', '--TEMPERATURE', default=0.1, type=float,
                        help="temperature to use during contrastive loss calculation")
    parser.add_argument('-c', '--CALIFORNIA', default='False', type=str,
                        help="are you running with california data")
    args = parser.parse_args()

    arch_dict = {'ResNet50': ResNet50,
                 'ResNet101V2':ResNet101V2,
                 'Xception':Xception,
                 'InceptionV3':InceptionV3}
    ca_flag_dict = {'True':True, 'False':False}
        
    run_model(args.output,
                  BATCH_SIZE=args.BATCH_SIZE,
                  epochs=args.EPOCHS,
                  architecture=arch_dict[args.arch],
                  temperature=args.TEMPERATURE,
                  ca_flag=ca_flag_dict[args.CALIFORNIA])

