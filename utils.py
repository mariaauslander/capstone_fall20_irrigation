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

def hyperspectral_image_generator(files, class_indices, batch_size=32, image_mean=None,
                           rotation_range=0, shear_range=0, scale_range=1,
                           transform_range=0, horizontal_flip=False,
                           vertical_flip=False, crop=False, crop_size=None, filling_mode='edge',
                           speckle_noise=None):
    from skimage.io import imread
    import numpy as np
    from random import sample
    from image_functions import categorical_label_from_full_file_name, preprocessing_image_ms

    while True:
        # select batch_size number of samples without replacement
        batch_files = sample(files, batch_size)
        # get one_hot_label
        batch_Y = categorical_label_from_full_file_name(batch_files,
                                                        class_indices)
        # array for images
        batch_X = []
        # loop over images of the current batch
        for idx, input_path in enumerate(batch_files):
            image = np.array(imread(input_path), dtype=float)
            if image_mean is not None:
                mean_std_data = np.loadtxt(image_mean, delimiter=',')
                image = preprocessing_image_ms(image, mean_std_data[0], mean_std_data[1])
            # process image
            image = augmentation_image_ms(image, rotation_range=rotation_range, shear_range=shear_range,
                                          scale_range=scale_range,
                                          transform_range=transform_range, horizontal_flip=horizontal_flip,
                                          vertical_flip=vertical_flip, warp_mode=filling_mode)
            if speckle_noise is not None:
                from skimage.util import random_noise
                image_max = np.max(np.abs(image), axis=(0, 1))
                image /= image_max

                image = random_noise(image, mode='speckle', var=speckle_noise)
                image *= image_max

            if crop:
                if crop_size is None:
                    crop_size = image.shape[0:2]
                image = crop_image(image, crop_size)
            # put all together
            batch_X += [image]
        # convert lists to np.array
        X = np.array(batch_X)
        Y = np.array(batch_Y)

        yield(X, Y)


def hyperspectral_image_generator_jp2(files, shape_file, class_indices_column, batch_size=32, image_mean=None,
                                      rotation_range=0, shear_range=0, scale_range=1,
                                      transform_range=0, horizontal_flip=False,
                                      vertical_flip=False, crop_size=None, filling_mode='edge',
                                      speckle_noise=None):
    from rasterio.mask import mask
    from rasterio import open
    from shapely.geometry import box
    import geopandas as gpd
    import numpy as np
    from random import sample
    from image_functions import categorical_label_from_full_file_name, preprocessing_image_ms
    from keras.utils import to_categorical

    geometry_df = gpd.read_file(shape_file)
    centroids = geometry_df['geometry'].values
    class_indices = geometry_df[class_indices_column].values.astype(int)
    number_of_classes = class_indices.max()
    files_centroids = list(zip(files*len(centroids), list(centroids)*len(files), list(class_indices)*len(files)))
    while True:
        # select batch_size number of samples without replacement
        batch_files = sample(files_centroids, batch_size)
        # get one_hot_label

        batch_Y = []
        # array for images
        batch_X = []
        # loop over images of the current batch
        for idx, (rf, polycenter, label) in enumerate(batch_files):
            raster_file = open(rf)
            mask_polygon = box(max(polycenter.coords.xy[0][0] - raster_file.transform[0] * crop_size[0] * 2,
                                   raster_file.bounds.left),
                               max(polycenter.coords.xy[1][0] - raster_file.transform[4] * crop_size[1] * 2,
                                   raster_file.bounds.bottom),
                               min(polycenter.coords.xy[0][0] + raster_file.transform[0] * crop_size[0] * 2,
                                   raster_file.bounds.right),
                               min(polycenter.coords.xy[1][0] + raster_file.transform[4] * crop_size[1] * 2,
                                   raster_file.bounds.top))
            image, out_transform = mask(raster_file, shapes=[mask_polygon], crop=True, all_touched=True)
            image = np.transpose(image, (1, 2, 0))
            if image_mean is not None:
                mean_std_data = np.loadtxt(image_mean, delimiter=',')
                image = preprocessing_image_ms(image.astype(np.float64), mean_std_data[0], mean_std_data[1])
            # process image
            image = augmentation_image_ms(image, rotation_range=rotation_range, shear_range=shear_range,
                                          scale_range=scale_range,
                                          transform_range=transform_range, horizontal_flip=horizontal_flip,
                                          vertical_flip=vertical_flip, warp_mode=filling_mode)
            if speckle_noise is not None:
                from skimage.util import random_noise
                image_max = np.max(np.abs(image), axis=(0, 1))
                image /= image_max

                image = random_noise(image, mode='speckle', var=speckle_noise)
                image *= image_max

            image = crop_image(image, crop_size)

            # put all together
            batch_X += [image]
            batch_Y += [to_categorical(label, num_classes=number_of_classes)]
        # convert lists to np.array
        X = np.array(batch_X)
        Y = np.array(batch_Y)

        yield(X, Y)


def augmentation_image_ms(image, rotation_range=0, shear_range=0, scale_range=1, transform_range=0,
                          horizontal_flip=False, vertical_flip=False, warp_mode='edge'):
    from skimage.transform import AffineTransform, SimilarityTransform, warp
    from numpy import deg2rad, flipud, fliplr
    from numpy.random import uniform, random_integers
    from random import choice

    image_shape = image.shape
    # Generate image transformation parameters
    rotation_angle = uniform(low=-abs(rotation_range), high=abs(rotation_range))
    shear_angle = uniform(low=-abs(shear_range), high=abs(shear_range))
    scale_value = uniform(low=abs(1 / scale_range), high=abs(scale_range))
    translation_values = (random_integers(-abs(transform_range), abs(transform_range)),
                          random_integers(-abs(transform_range), abs(transform_range)))

    # Horizontal and vertical flips
    if horizontal_flip:
        # randomly flip image up/down
        if choice([True, False]):
            image = flipud(image)
    if vertical_flip:
        # randomly flip image left/right
        if choice([True, False]):
            image = fliplr(image)

    # Generate transformation object
    transform_toorigin = SimilarityTransform(scale=(1, 1), rotation=0, translation=(-image_shape[0], -image_shape[1]))
    transform_revert = SimilarityTransform(scale=(1, 1), rotation=0, translation=(image_shape[0], image_shape[1]))
    transform = AffineTransform(scale=(scale_value, scale_value), rotation=deg2rad(rotation_angle),
                                shear=deg2rad(shear_angle), translation=translation_values)
    # Apply transform
    image = warp(image, ((transform_toorigin) + transform) + transform_revert, mode=warp_mode, preserve_range=True)
    return image


def crop_image(image, target_size):
    from numpy import ceil, floor
    x_crop = min(image.shape[0], target_size[0])
    y_crop = min(image.shape[1], target_size[1])
    midpoint = [ceil(image.shape[0] / 2), ceil(image.shape[1] / 2)]

    out_img = image[int(midpoint[0] - ceil(x_crop / 2)):int(midpoint[0] + floor(x_crop / 2)),
              int(midpoint[1] - ceil(y_crop / 2)):int(midpoint[1] + floor(y_crop / 2)),
              :]
    assert list(out_img.shape[0:2]) == target_size
    return out_img
