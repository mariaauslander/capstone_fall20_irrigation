import tensorflow as tf
import cv2
import numpy as np
import time
from tensorflow.keras.preprocessing import image

datafile_image_patches = {}

# Reades TF Record which includes additional parameters
def read_tfrecord_extra_info(example):
    return read_tfrecord(example, return_extra_info=True)

def read_tfrecord_multi_classes(example):
    return read_tfrecord(example, multi=True)

def read_tfrecord(example, return_extra_info=False, multi=False):
    '''
    THIS FUNCTION IS USED TO PARSE THE TFRECORDS FILES FOR BIGEARTHNET DATA.
    THE BAND STATISTICS WERE PROVIDED BY THE BIGEARTHNET TEAM
    '''
    BAND_STATS = {
        'mean': {
            'B01': 340.76769064,
            'B02': 429.9430203,
            'B03': 614.21682446,
            'B04': 590.23569706,
            'B05': 950.68368468,
            'B06': 1792.46290469,
            'B07': 2075.46795189,
            'B08': 2218.94553375,
            'B8A': 2266.46036911,
            'B09': 2246.0605464,
            'B11': 1594.42694882,
            'B12': 1009.32729131
        },
        'std': {
            'B01': 554.81258967,
            'B02': 572.41639287,
            'B03': 582.87945694,
            'B04': 675.88746967,
            'B05': 729.89827633,
            'B06': 1096.01480586,
            'B07': 1273.45393088,
            'B08': 1365.45589904,
            'B8A': 1356.13789355,
            'B09': 1302.3292881,
            'B11': 1079.19066363,
            'B12': 818.86747235
        }
    }

    # Use this one-liner to standardize each feature prior to reshaping.
    def standardize_feature(data, band_name):
        '''
        APPLY STANDARDIZATION AND SCALING CONSISTENT WITH BEN PROCEDURE
        '''
        return ((tf.dtypes.cast(data, tf.float32) - BAND_STATS['mean'][band_name]) / BAND_STATS['std'][band_name])

    # decode the TFRecord
    # The parse single example methods takes an example (from a tfrecords file),
    # and a dictionary that explains the data format of each feature.
    example = tf.io.parse_single_example(example, {
        'B01': tf.io.FixedLenFeature([20 * 20], tf.int64),
        'B02': tf.io.FixedLenFeature([120 * 120], tf.int64),
        'B03': tf.io.FixedLenFeature([120 * 120], tf.int64),
        'B04': tf.io.FixedLenFeature([120 * 120], tf.int64),
        'B05': tf.io.FixedLenFeature([60 * 60], tf.int64),
        'B06': tf.io.FixedLenFeature([60 * 60], tf.int64),
        'B07': tf.io.FixedLenFeature([60 * 60], tf.int64),
        'B08': tf.io.FixedLenFeature([120 * 120], tf.int64),
        'B8A': tf.io.FixedLenFeature([60 * 60], tf.int64),
        'B09': tf.io.FixedLenFeature([20 * 20], tf.int64),
        'B11': tf.io.FixedLenFeature([60 * 60], tf.int64),
        'B12': tf.io.FixedLenFeature([60 * 60], tf.int64),
        'patch_name': tf.io.VarLenFeature(dtype=tf.string),
        'original_labels': tf.io.VarLenFeature(dtype=tf.string),
        'original_labels_multi_hot': tf.io.FixedLenFeature([43], tf.int64)
    })

    # https://gitlab.tubit.tu-berlin.de/rsim/BigEarthNet-S2_43-classes_models/blob/master/label_indices.json
    # "Permanently irrigated land": 12,
    example['binary_label'] = example['original_labels_multi_hot'][tf.constant(12)]

    # After parsing our data into a tensor, let's standardize and reshape.
    reshaped_example = {
        'B01': tf.reshape(standardize_feature(example['B01'], 'B01'), [20, 20]),
        'B02': tf.reshape(standardize_feature(example['B02'], 'B02'), [120, 120]),
        'B03': tf.reshape(standardize_feature(example['B03'], 'B03'), [120, 120]),
        'B04': tf.reshape(standardize_feature(example['B04'], 'B04'), [120, 120]),
        'B05': tf.reshape(standardize_feature(example['B05'], 'B05'), [60, 60]),
        'B06': tf.reshape(standardize_feature(example['B06'], 'B06'), [60, 60]),
        'B07': tf.reshape(standardize_feature(example['B07'], 'B07'), [60, 60]),
        'B08': tf.reshape(standardize_feature(example['B08'], 'B08'), [120, 120]),
        'B8A': tf.reshape(standardize_feature(example['B8A'], 'B8A'), [60, 60]),
        'B09': tf.reshape(standardize_feature(example['B09'], 'B09'), [20, 20]),
        'B11': tf.reshape(standardize_feature(example['B11'], 'B11'), [60, 60]),
        'B12': tf.reshape(standardize_feature(example['B12'], 'B12'), [60, 60]),
        'patch_name': example['patch_name'],
        'original_labels': example['original_labels'],
        'original_labels_multi_hot': example['original_labels_multi_hot'],
        'binary_labels': example['binary_label']
    }

    # Next sort the layers by resolution
    bands_10m = tf.stack([reshaped_example['B04'],
                          reshaped_example['B03'],
                          reshaped_example['B02'],
                          reshaped_example['B08']], axis=2)

    bands_20m = tf.stack([reshaped_example['B05'],
                          reshaped_example['B06'],
                          reshaped_example['B07'],
                          reshaped_example['B8A'],
                          reshaped_example['B11'],
                          reshaped_example['B12']], axis=2)

    # Finally resize the 20m data and stack the bands together.
    img = tf.concat([bands_10m, tf.image.resize(bands_20m, [120, 120], method='bicubic')], axis=2)
    
    multi_hot_label = reshaped_example['original_labels_multi_hot']
    binary_label = reshaped_example['binary_labels']

    # Can update this to return the multi-label if doing multi-class classification
    if multi:
        return img, multi_hot_label
    else:
        if return_extra_info:
            return img, binary_label, reshaped_example['patch_name']
        else:
            return img, binary_label

def read_ca_tfrecord(example):
    '''
    THE CALIFORNIA DATA HAS DIFFERENT POPULATION STATISTICS AS EXPECETED.
    CALCULATED VIA THE process_california_data.ipynb file
    '''
    BAND_STATS = {'mean': {'B02': 745.8342280288207,
                          'B03': 1066.1362867829712,
                          'B04': 1294.678473044234,
                          'B05': 1645.7598649250806,
                          'B06': 2246.824426424665,
                          'B07': 2516.3336991935817,
                          'B08': 2688.8463771950937,
                          'B8A': 2733.816949232295,
                          'B11': 2769.942382613557,
                          'B12': 2092.625560070325},
                   'std': {'B02': 504.9172431483328,
                          'B03': 616.4692423335321,
                          'B04': 851.3811496920607,
                          'B05': 795.0872173538605,
                          'B06': 765.746057996193,
                          'B07': 871.266391942569,
                          'B08': 919.4293720949656,
                          'B8A': 891.7677760562052,
                          'B11': 1083.5092422778923,
                          'B12': 1101.34386721669}}

    # Use this one-liner to standardize each feature prior to reshaping.
    def standardize_feature(data, band_name):
        return ((tf.dtypes.cast(data, tf.float32) - BAND_STATS['mean'][band_name]) / BAND_STATS['std'][band_name])

    # decode the TFRecord
    # The parse single example methods takes an example (from a tfrecords file),
    # and a dictionary that explains the data format of each feature.
    example = tf.io.parse_single_example(example, {
        'B02': tf.io.FixedLenFeature([120 * 120], tf.int64),
        'B03': tf.io.FixedLenFeature([120 * 120], tf.int64),
        'B04': tf.io.FixedLenFeature([120 * 120], tf.int64),
        'B05': tf.io.FixedLenFeature([120 * 120], tf.int64),
        'B06': tf.io.FixedLenFeature([120 * 120], tf.int64),
        'B07': tf.io.FixedLenFeature([120 * 120], tf.int64),
        'B08': tf.io.FixedLenFeature([120 * 120], tf.int64),
        'B8A': tf.io.FixedLenFeature([120 * 120], tf.int64),
        'B11': tf.io.FixedLenFeature([120 * 120], tf.int64),
        'B12': tf.io.FixedLenFeature([120 * 120], tf.int64)
    })

    # After parsing our data into a tensor, let's standardize and reshape.
    reshaped_example = {
        'B02': tf.reshape(standardize_feature(example['B02'], 'B02'), [120, 120]),
        'B03': tf.reshape(standardize_feature(example['B03'], 'B03'), [120, 120]),
        'B04': tf.reshape(standardize_feature(example['B04'], 'B04'), [120, 120]),
        'B05': tf.reshape(standardize_feature(example['B05'], 'B05'), [120, 120]),
        'B06': tf.reshape(standardize_feature(example['B06'], 'B06'), [120, 120]),
        'B07': tf.reshape(standardize_feature(example['B07'], 'B07'), [120, 120]),
        'B08': tf.reshape(standardize_feature(example['B08'], 'B08'), [120, 120]),
        'B8A': tf.reshape(standardize_feature(example['B8A'], 'B8A'), [120, 120]),
        'B11': tf.reshape(standardize_feature(example['B11'], 'B11'), [120, 120]),
        'B12': tf.reshape(standardize_feature(example['B12'], 'B12'), [120, 120])
    }

    # Next sort the layers by resolution - all the same resolution for CA
    bands_10m = tf.stack([reshaped_example['B04'],
                          reshaped_example['B03'],
                          reshaped_example['B02'],
                          reshaped_example['B08']], axis=2)

    bands_20m = tf.stack([reshaped_example['B05'],
                          reshaped_example['B06'],
                          reshaped_example['B07'],
                          reshaped_example['B8A'],
                          reshaped_example['B11'],
                          reshaped_example['B12']], axis=2)

    # Finally resize the 20m data and stack the bands together.
    img = tf.concat([bands_10m, bands_20m], axis=2)
    
    return img, 0

def get_batched_dataset(filenames, batch_size, augment=False, simclr=False, ca=False, shuffle=False, num_classes=1):
    '''
    This function is used to return a batch generator for training our tensorflow model.
    basically we read from different tfrecords files, and shuffle our records.
    we use the appropriate parsing function depending on if it is CA data or BigEarthNet data
    Finally - if it is a SimCLR model do not repeat the dataset, as we manually loop over our data
    and train our model in the simclr.py script.
    '''
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    dataset = tf.data.Dataset.list_files(filenames, shuffle=shuffle)
    print(f'Filenames: {filenames}')
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=2, num_parallel_calls=1)

    # [todo] shuffle after map and caching? https://www.tensorflow.org/datasets/keras_example
    if simclr:
        dataset = dataset.shuffle(buffer_size=2048, reshuffle_each_iteration=False)
    else:
        # dataset = dataset.shuffle(buffer_size=2048, reshuffle_each_iteration=False).repeat()
        dataset = dataset.shuffle(buffer_size=2048).repeat()

    #Cache the initial dataset
    dataset_for_patches = dataset

    if ca:
      dataset = dataset.map(read_ca_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        if num_classes > 1:
            dataset = dataset.map(read_tfrecord_multi_classes, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            dataset = dataset.map(read_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)


    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)  #

    # The above dataset doesn't have the real data loaded when the map function is called.
    # Using take function on 2 examples to minimize the performance issues and side effects
    # global datafile_image_patches
    # image_patches = []
    # dataset_examples = dataset_for_patches.map(read_tfrecord_extra_info,
    #                                            num_parallel_calls=tf.data.experimental.AUTOTUNE).take(2)
    # for img, label, patch in dataset_examples:
    #     image_patches.append(str(patch.values.numpy())[3:-2])
    # datafile_image_patches[filenames] = image_patches
    #print(datafile_image_patches)
    return dataset

class TimeHistory(tf.keras.callbacks.Callback):
  
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

class Augment():
  def augfunc(self, sample):        
    # Randomly apply transformation (color distortions) with probability p.
    sample = self._random_apply(self._color_jitter, sample, p=0.8)
    sample = self._random_apply(self._color_drop, sample, p=0.2)
    sample = self._random_apply(self._blur, sample, p=0.5)

    return sample

  def _color_jitter(self,  x, s=0.50):
      # one can also shuffle the order of following augmentations
      # each time they are applied.
      x = tf.image.random_brightness(x, max_delta=0.8*s)
      x = tf.image.random_contrast(x, lower=1-0.8*s, upper=1+0.8*s)
      dx = tf.image.random_saturation(x[:,:,:3], lower=1-0.8*s, upper=1+0.8*s)
      dx = tf.image.random_hue(dx, max_delta=0.2*s)
      x = tf.concat([dx, x[:,:,3:]],axis=2)
      x = tf.clip_by_value(x, 0, 1)
      return x

  def _color_drop(self, x):
      dx = tf.image.rgb_to_grayscale(x[:,:,:3])
      dx = tf.tile(dx, [1, 1, 3])
      x = tf.concat([dx, x[:,:,3:]],axis=2)
      return x

  def _blur(self, x):
      # SimClr implementation is applied at 10% of image size with a random sigma
      p = np.random.uniform(0.1,2)
      if type(x) == np.ndarray:
          return (cv2.GaussianBlur(x,(5,5),p))
      return (cv2.GaussianBlur(x.numpy(),(5,5),p))

  def _random_apply(self, func, x, p):
      return tf.cond(
        tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
                tf.cast(p, tf.float32)),
        lambda: func(x),
        lambda: x)
