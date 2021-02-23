import os
import pandas as pd
import tensorflow as tf
from glob import glob
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
#from augmentation.gaussian_filter import GaussianBlur
# from utils import *
import helpers
import losses
import argparse
import cv2
import wandb

import sys
# import local helpers
sys.path.append('/workspace/app/src')

from data.dataset_helper import *
import params

# print(f'Using TensorFlow Version: {tf.__version__}')

# Create model logs folder
if not os.path.exists('model_logs'):
    os.makedirs('model_logs')


def build_simclr_model(imported_model, hidden_1, hidden_2, hidden_3):
    """
    1. This function is used to create the neural encoder and projection head. 
    2. The neural encoder arch can be chosen from {ResNet50, ResNet101V2, Xception, InceptionV3}.
    3. Training 10 channels(BigEarthNet) of the satellite data. The projection head dimensions should be specified as inputs.

    4. imported_model: tensorflow.keras.applications model - encoder arch(eg. ResNet101V2 is typically used).
    hidden_1: integer - dimension of the first layer, the projection head.
    hidden_2: integer - dimension of the second layer, the projection head.
    hidden_3: integer - output dimension - vector used in the contrastive loss function
    """

    # Load in a Keras Model for our neural encoder and set to trainable
    base_model = imported_model(include_top=False, weights=None, input_shape=[120, 120, 10])
    base_model.trainable = True

    # Input dimensions are fixed to BigEarthNet image dimensions.
    inputs = Input((120, 120, 10))

    # Add a Global Average Pooling layer to flatten the output of the neural encoder
    h = base_model(inputs, training=True)
    h = GlobalAveragePooling2D()(h)

    # Add the projection head layers with `relu` activations
    projection_1 = Dense(hidden_1)(h)
    projection_1 = Activation('relu')(projection_1)
    projection_2 = Dense(hidden_2)(projection_1)
    projection_2 = Activation('relu')(projection_2)
    projection_3 = Dense(hidden_3)(projection_2)

    # Define the final model - SimCLR
    simclr_model = tf.keras.models.Model(inputs, projection_3)

    return simclr_model

# Enable eager execution
tf.config.run_functions_eagerly(True)
# remove warning from keras models
tf.get_logger().setLevel('ERROR')

@tf.function
def train_step(xis, xjs, model, optimizer, criterion, temperature, batch_size):
    
    # Mask that remove positive examples from the batch of negative samples
    negative_mask = helpers.get_negative_mask(batch_size)
  
    with tf.GradientTape() as tape:
        # Get the latent space vectors for our pairs of augmented images
        zis = model(xis)
        zjs = model(xjs)

        # Normalize projection feature vectors
        zis = tf.math.l2_normalize(zis, axis=1)
        zjs = tf.math.l2_normalize(zjs, axis=1)

        # Similarity between all positive pairs
        l_pos = losses._dot_simililarity_dim1(zis, zjs)
        l_pos = tf.reshape(l_pos, (batch_size, 1))
        
        # Divide by the temperature variable `tau`
        l_pos /= temperature
        
        # Combine all images to create negative array 
        negatives = tf.concat([zjs, zis], axis=0)

        loss = 0
        # Compare each image vector to every other image vector 
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


def run_model(BATCH_SIZE, epochs, architecture, temperature, ca_flag):
    
    '''
    Main execution function used to take input flags and control the model flow.
    
    name: -string Output name for model file
    BATCH_SIZE: int- batch size to use during training - set to be large
    epochs: int - number of passes over the data
    architecture: - tensorflow.keras.applications model to use as neural encoder
    temperature: float - temperature for the softmax
    ca_flag: Boolean - specify whether training on California data or BEN data
    '''

    # California data has different files
    if ca_flag:
        training_filenames = os.path.join(params.TFR_PATH, "original", params.IMBALANCED_CA_TRAINING_FILENAMES)
    else:
        training_filenames = os.path.join(params.TFR_PATH, "original", params.IMBALANCED_TRAINING_FILENAMES)

    # Get the training files in batches

    training_data = get_batched_dataset(training_filenames, batch_size=128, shuffle=False,
                                    num_classes=1, simclr=True, ca=ca_flag)

    # Use Cross Entropy Loss
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                              reduction=tf.keras.losses.Reduction.SUM
                                                             )
    
    # Learning Rate Decay with stochastic gradient descent `SGD`
    decay_steps = 1000
    lr_decayed_fn = tf.keras.experimental.CosineDecay(
        initial_learning_rate=0.1, 
        decay_steps=decay_steps
    )
    optimizer = tf.keras.optimizers.SGD(lr_decayed_fn)

    # Build the model with following hidden layer sizes
    simclr_2 = build_simclr_model(architecture, 1024, 512, 128)
    
    # Model summary
    simclr_2.summary()

    # create a list for tracking losses by epoch
    epoch_wise_loss = []
    
    # Track time spent per epoch
    time_callback = TimeHistory()
    
    # Augment Class used for color distortion and Gaussian Blur
    augment = Augment()
    
    # Set other data augmentation
    # [todo change to command parameters]
    ROTATION = 180
    SHIFT = 0.10
    FLIP = True
    ZOOM = 0.20
    JITTER = 0.0
    BLUR = True
    
    # Use Keras image preprocessing to augment images in batches
    datagen = image.ImageDataGenerator(
        rotation_range=ROTATION,
        width_shift_range=SHIFT,
        height_shift_range=SHIFT,
        horizontal_flip=FLIP,
        vertical_flip=FLIP,
        zoom_range=ZOOM,
        preprocessing_function= augment.augfunc
    )

    min_loss = 1e6
    min_loss_epoch = 0

    # Loop through epochs and batches
    for epoch in tqdm(range(epochs)):
        step_wise_loss = []
        
        # Loop over batches, perform augmentation and calculate poss
        for image_batch in tqdm(training_data):
            # Use the data generator to augment the data - DO NOT SHUFFLE - images need to stay aligned
            a = datagen.flow(image_batch, batch_size=BATCH_SIZE, shuffle=False)
            b = datagen.flow(image_batch, batch_size=BATCH_SIZE, shuffle=False)
            
            # Send image arrays, simclr model, etc to our train_step function
            loss = train_step(a[0][0], b[0][0], simclr_2,
                              optimizer,
                              criterion,
                              temperature=temperature,
                              batch_size=BATCH_SIZE
                             )
            step_wise_loss.append(loss)

            ###### log loss to wandb
            wandb.log({"InfoNCE": loss})
      
        # Append to list of loss by epoch
        epoch_wise_loss.append(np.mean(step_wise_loss))

        # ###### log plot to wandb
        # fig, ax = plt.subplots()
        # ax.plot(step_wise_loss)
        # ax.set_ylabel("loss")
        # # Log the plot
        # wandb.log({"plot": fig})
        # fig
        
        # Print the loss after every epoch
        print(f"****epoch: {epoch + 1} loss: {epoch_wise_loss[-1]:.3f}****\n")
        
        # Save weights every five epochs
        # if (epoch > 0) and ((epoch+1) % 5 == 0):
        #     print(f'Saving weights for epoch: {epoch+1}')
            
            # Save the final model with weights
            # simclr_2.save(f'{OUTPUT_PATH}/{name}_{epoch+1}.h5')

            ###### save model to wandb
            # simclr_2.save(f'{wandb.run.dir}/{name}_{epoch+1}.h5')

    simclr_2.save(os.path.join(wandb.run.dir, "model.h5"))

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

    #
    print(df)

    ###### wandb
    wandb.run.summary["epoch_wise_loss"] = epoch_wise_loss[0]
    wandb.run.summary["rotation"] = ROTATION
    wandb.run.summary["shift"] = SHIFT
    wandb.run.summary["flip"] = FLIP
    wandb.run.summary["zoom"] = ZOOM
    wandb.run.summary["jitter"] = JITTER
    wandb.run.summary["blur"] = BLUR

    #
    # df.to_pickle(f'{OUTPUT_PATH}/{name}.pkl')
    
    # return df
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

    print('In main function')
    parser = argparse.ArgumentParser(description='Script for running different supervised classifiers')
    parser.add_argument('-a', '--architecture', choices=['ResNet50', 'ResNet101V2', 'Xception', 'InceptionV3'],
                        help='Class of Model Architecture to use for classification')
    # parser.add_argument('-o', '--output', type=str,
    #                     help='Output File Prefix for model file and dataframe')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                       help="batch size to use during training and validation")
    parser.add_argument('-e', '--epochs', default=50, type=int,
                        help="number of epochs to run")
    parser.add_argument('-t', '--temperature', default=0.1, type=float,
                        help="temperature to use during contrastive loss calculation")
    # parser.add_argument('-c', '--california', default=False, type=str2bool,
    #                     help="are you running with california data")
    args = parser.parse_args()

    ###### w&b run
    wandb.init(project="SimCLR_BigEarthNet", entity="cal-capstone")
    wandb.config.update(args)  # adds all of the arguments as config variables
    wandb.config.update({'framework': f'TensorFlow {tf.__version__}'})

    arch_dict = {'ResNet50': ResNet50,
                 'ResNet101V2':ResNet101V2,
                 'Xception':Xception,
                 'InceptionV3':InceptionV3}
    
    ca_flag_dict = {'True':True, 'False':False}
        
    run_model(BATCH_SIZE=args.batch_size,
              epochs=args.epochs,
              architecture=arch_dict[args.architecture],
              temperature=args.temperature,
              ca_flag=False
             )

