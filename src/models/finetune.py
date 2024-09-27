import pandas as pd
import tensorflow as tf
import os
from tensorflow.keras.layers import *
from src.data.dataset_helper import *
import argparse

from model_helper import *

# Set Paths

BASE_PATH = './BigEarthData'
OUTPUT_PATH = os.path.join(BASE_PATH, 'models')
TFR_PATH = os.path.join(BASE_PATH, 'tfrecords')

# Use the following metrics for evaluation

def finetune_pretrained_model(model, num_unfrozen, metrics=METRICS):
  '''
  This function is used to simply finetune from the existing projection head, as opposed
  to stacking a new MLP on top of a projection head output as is done above.
  '''
  pretrained_model = tf.keras.models.load_model(model)
  
  # Freeze all layers
  pretrained_model.trainable = False 
  
  # Unfreeze just the projection head
  for layer in pretrained_model.layers[-num_unfrozen:]:
      layer.trainable=True

  # Add output layer
  output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(pretrained_model.layers[-1].output)
  
  # define new model
  new_model = tf.keras.models.Model(inputs=pretrained_model.inputs, outputs=output)

  ft_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)  

  # Compile model with Cross Entropy loss
  new_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=ft_optimizer,
              metrics=metrics)

  return new_model


def run_model(name, pretrained_model, BATCH_SIZE, epochs, training_dataset, CLASS, NUM_UNFROZEN):
    print(50 * "*")
    print(f"Running model: {name}")
    print(50 * "=")
    print(f"Batch Size: {BATCH_SIZE}")

    training_filenames = f'{TFR_PATH}/{training_dataset}'
    
    if CLASS == 'Vineyards':
      validation_filenames = f'{TFR_PATH}/final_balanced_val_vy.tfrecord'
    else:
      validation_filenames = f'{TFR_PATH}/balanced_val.tfrecord'

    training_data = get_batched_dataset(training_filenames, batch_size=BATCH_SIZE, shuffle=True)
    val_data = get_batched_dataset(validation_filenames, batch_size=BATCH_SIZE, shuffle=False)

    len_val_records = 4820 
    if training_dataset == 'final_balanced_train_10percent.tfrecord':
      len_train_records = 1024
    elif training_dataset == 'final_balanced_train_3percent.tfrecord':
      len_train_records = 256
    elif training_dataset == 'final_balanced_train_vy_3percent.tfrecord':
      len_train_records = 256
    elif training_dataset == 'final_balanced_train_1percent.tfrecord':
      len_train_records = 64
    elif training_dataset == 'final_balanced_train_vy_1percent.tfrecord':
      len_train_records = 64
    elif training_dataset == 'balanced_train_0.tfrecord':
      len_train_records = 9942
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
    if NUM_UNFROZEN:
      print(f'Finetuning the SimCLR model from the {(NUM_UNFROZEN+1)//2} layer of the projection head')
      model = finetune_pretrained_model(pretrained_model, NUM_UNFROZEN)
    else:
      print(f'Adding new MLP to second layer of Projection head')
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
      

    return df.val_auc.max()
  
if __name__ == '__main__':
    
    print('In main function')
    parser = argparse.ArgumentParser(description='Script for running different supervised classifiers')
    parser.add_argument('-p', '--pretrained',
                        choices=[file for file in os.listdir('./BigEarthData/models/') if file[-3:]=='.h5'],
                        help='Which pretrained model do you want to finetune?')
    parser.add_argument('-o', '--output', type=str,
                        help='Output File Prefix for model file and dataframe')
    parser.add_argument('-b', '--BATCH_SIZE', default=32, type=int,
                       help="batch size to use during training and validation")
    parser.add_argument('-e', '--EPOCHS', default=10, type=int,
                        help="number of epochs to run")
    parser.add_argument('-c', '--CLASS', default='Irrigation', type=str,
                        help="which class to finetune on", choices=['Irrigation', 'Vineyards'])
    parser.add_argument('-u', '--UNFROZEN', default=None, type=int,
                        help="Number of layers of PH to unfreeze during finetuning. If none, will add new MLP ontop of second PH layer", choices=[1, 3, 5])
    parser.add_argument('-t', '--training_data', type=str,
                        choices=['final_balanced_train_10percent.tfrecord',
                                 'final_balanced_train_3percent.tfrecord',
                                 'final_balanced_train_1percent.tfrecord',
                                 'final_balanced_train_vy_3percent.tfrecord',
                                 'final_balanced_train_vy_1percent.tfrecord',
                                 'balanced_train_0.tfrecord',
                                 'balanced_train_*'])
    
    args = parser.parse_args()
    
    
    print(f'Using TensorFlow Version: {tf.__version__}')


    
    model_path = os.path.join(OUTPUT_PATH, args.pretrained)

    best_score = run_model(name=args.output,
              pretrained_model=model_path,
              BATCH_SIZE=args.BATCH_SIZE,
              epochs=args.EPOCHS,
              training_dataset=args.training_data,
              CLASS = args.CLASS,
              NUM_UNFROZEN = args.UNFROZEN)
    
    print(f'Best Score: {best_score}')
    
    
