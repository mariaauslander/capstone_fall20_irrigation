import tensorflow as tf
import numpy as np
import os
import json

# Spectral band names to read related GeoTIFF files
band_names = ['B01', 'B02', 'B03', 'B04', 'B05',
              'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']

def prep_example(bands, original_labels, original_labels_multi_hot, patch_name):
    return tf.train.Example(
            features=tf.train.Features(
                feature={
                    'B01': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=np.ravel(bands['B01']))),
                    'B02': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=np.ravel(bands['B02']))),
                    'B03': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=np.ravel(bands['B03']))),
                    'B04': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=np.ravel(bands['B04']))),
                    'B05': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=np.ravel(bands['B05']))),
                    'B06': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=np.ravel(bands['B06']))),
                    'B07': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=np.ravel(bands['B07']))),
                    'B08': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=np.ravel(bands['B08']))),
                    'B8A': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=np.ravel(bands['B8A']))),
                    'B09': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=np.ravel(bands['B09']))),
                    'B11': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=np.ravel(bands['B11']))),
                    'B12': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=np.ravel(bands['B12']))),
                    'original_labels': tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[i.encode('utf-8') for i in original_labels])),
                    'original_labels_multi_hot': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=original_labels_multi_hot)),
                    'patch_name': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[patch_name.encode('utf-8')]))
                }))

def create_split(root_folder, patch_names, TFRecord_writer, label_indices, GDAL_EXISTED, RASTERIO_EXISTED):
    if GDAL_EXISTED:
        import gdal
    elif RASTERIO_EXISTED:
        import rasterio
    progress_bar = tf.keras.utils.Progbar(target = len(patch_names))
    for patch_idx, patch_name in enumerate(patch_names):
        patch_folder_path = os.path.join(root_folder, patch_name)
        bands = {}
        for band_name in band_names:
            # First finds related GeoTIFF path and reads values as an array
            band_path = os.path.join(
                patch_folder_path, patch_name + '_' + band_name + '.tif')
            if GDAL_EXISTED:
                band_ds = gdal.Open(band_path,  gdal.GA_ReadOnly)
                raster_band = band_ds.GetRasterBand(1)
                band_data = raster_band.ReadAsArray()
                bands[band_name] = np.array(band_data)
            elif RASTERIO_EXISTED:
                band_ds = rasterio.open(band_path)
                band_data = np.array(band_ds.read(1))
                bands[band_name] = np.array(band_data)
        
        original_labels_multi_hot = np.zeros(
            len(label_indices['original_labels'].keys()), dtype=int)
        patch_json_path = os.path.join(
            patch_folder_path, patch_name + '_labels_metadata.json')

        with open(patch_json_path, 'rb') as f:
            patch_json = json.load(f)

        original_labels = patch_json['labels']
        for label in original_labels:
            original_labels_multi_hot[label_indices['original_labels'][label]] = 1
        
        example = prep_example(
            bands, 
            original_labels,
            original_labels_multi_hot,
            patch_name
        )
        TFRecord_writer.write(example.SerializeToString())
        progress_bar.update(patch_idx)

def prep_tf_record_files(root_folder, out_folder, split_names, patch_names_list, label_indices, GDAL_EXISTED, RASTERIO_EXISTED):
    try:
        writer_list = []
        for split_name in split_names:
            writer_list.append(
                    tf.io.TFRecordWriter(os.path.join(
                        out_folder, split_name + '.tfrecord'))
                )
    except:
        print('ERROR: TFRecord writer is not able to write files')
        return

    for split_idx in range(len(patch_names_list)):
        print('INFO: creating the split of', split_names[split_idx], 'is started')
        create_split(
            root_folder, 
            patch_names_list[split_idx], 
            writer_list[split_idx],
            label_indices,
            GDAL_EXISTED, 
            RASTERIO_EXISTED, 
            )
        writer_list[split_idx].close()



