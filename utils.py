import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image


def read_tfrecord(example):
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
    
    # Can update this to return the multilabel if doing multi-class classification
    return img, binary_label
  
  
def get_batched_dataset(filenames, batch_size, augment=False):
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    dataset = tf.data.Dataset.list_files(filenames, shuffle=True)
    print(f'Filenames: {filenames}')
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=2, num_parallel_calls=1)
    dataset = dataset.shuffle(buffer_size=2048)
    #.repeat()
    
    dataset = dataset.map(read_tfrecord, num_parallel_calls=10)
    dataset = dataset.batch(batch_size, drop_remainder=False)  # drop_remainder will be needed on TPU
    dataset = dataset.prefetch(5)  #

    return dataset
