#!/usr/bin/env python
# coding: utf-8

import argparse
import csv
import json
import numpy as np
import os
import pandas as pd
import rasterio
import tensorflow as tf
from glob import glob
from tqdm import tqdm

# Path to the BigEarthNet extracted files
big_earth_path = '/workspace/app/data/raw/BigEarthNet-v1.0/'

# Models folder is already checkin. No need to download the models
big_earth_models_folder = '/workspace/app/data/raw/bigearthnet-models/'

# Stores the TFRecords
out_folder = '/workspace/app/data/processed'

if not os.path.exists(big_earth_path):
    print('ERROR: folder', big_earth_path, 'does not exist')

if not os.path.exists(big_earth_models_folder):
    print('ERROR: folder', big_earth_models_folder, 'does not exist')

if not os.path.exists(out_folder):
    print('ERROR: folder', out_folder, 'does not exist')

print(f'Using Python Version: {pd.__version__}')
print(f'Using TensorFlow Version: {tf.__version__}')

# Set up a symbolic link to allow for easy Python module imports. Then check to make sure the link works (it is a Unix link so check from shell)
os.system("ln -s '/workspace/app/data/raw/bigearthnet-models/' bemodels")
os.system('ls bemodels')
from bemodels import tensorflow_utils

# Downloads the data from teh Bigearthnet website
def download_data():
    os.system("curl http://bigearth.net/downloads/BigEarthNet-S2-v1.0.tar.gz -o data/raw/BigEarthNet-v1.0.zip")
    os.system("tar -xvf data/raw/BigEarthNet-v1.0.zip -C data/raw")

# Process All of the BigEarthNet data
def preprocess_tfrecords():
    with open(big_earth_models_folder + 'label_indices.json', 'rb') as f:
        label_indices = json.load(f)

    root_folder = big_earth_path

    csv_file_path_list = ['splits/train.csv', 'splits/test.csv', 'splits/val.csv']

    for csv_file in csv_file_path_list:
        splits = glob(f"{big_earth_models_folder}{csv_file}")
        patch_names_list = []
        split_names = []
        for csv_file in splits:
            patch_names_list.append([])
            split_names.append(os.path.basename(csv_file).split('.')[0])
            with open(csv_file, 'r') as fp:
                csv_reader = csv.reader(fp, delimiter=',')
                for row in csv_reader:
                    patch_names_list[-1].append(row[0].strip())
        tensorflow_utils.prep_tf_record_files(
            root_folder, out_folder,
            split_names, patch_names_list,
            label_indices, False, True)

    # Shard the Train data
    raw_dataset = tf.data.TFRecordDataset(out_folder + "/train.tfrecord")
    shards = 50
    for i in range(shards):
        writer = tf.data.experimental.TFRecordWriter(f"{out_folder}/train-part-{i}.tfrecord")
        writer.write(raw_dataset.shard(shards, i))

    # Shard the Test data
    raw_dataset = tf.data.TFRecordDataset(out_folder + "/test.tfrecord")
    shards = 20
    for i in range(shards):
        writer = tf.data.experimental.TFRecordWriter(f"{out_folder}/test-part-{i}.tfrecord")
        writer.write(raw_dataset.shard(shards, i))

    # Shard the Val data
    raw_dataset = tf.data.TFRecordDataset(out_folder + "/val.tfrecord")
    shards = 20
    for i in range(shards):
        writer = tf.data.experimental.TFRecordWriter(f"{out_folder}/val-part-{i}.tfrecord")
        writer.write(raw_dataset.shard(shards, i))


def preprocess_tfrecords_labelled():
    with open(big_earth_models_folder + 'label_indices.json', 'rb') as f:
        label_indices = json.load(f)

    root_folder = big_earth_path

    # splits = glob(f'/workspace/app/data/raw/bigearthnet-models/splits/val.csv')
    splits = glob(f'{big_earth_models_folder}splits/val.csv')

    # Checks the existence of patch folders and populate the list of patch folder paths
    folder_path_list = []
    if not os.path.exists(root_folder):
        print('ERROR: folder', root_folder, 'does not exist')

    patch_names_list = []
    split_names = []
    for csv_file in splits:
        patch_names_list.append([])
        split_names.append(os.path.basename(csv_file).split('.')[0])
        with open(csv_file, 'r') as fp:
            csv_reader = csv.reader(fp, delimiter=',')
            for row in csv_reader:
                patch_names_list[-1].append(row[0].strip())

    len(patch_names_list[0])

    irrigated_examples = []
    nonirrigated_examples = []
    missing_count = 0
    for patch_name in tqdm(patch_names_list[0]):
        patch_folder_path = os.path.join(root_folder, patch_name)
        patch_json_path = os.path.join(
            patch_folder_path, patch_name + '_labels_metadata.json')
        try:
            with open(patch_json_path, 'rb') as f:
                patch_json = json.load(f)
        except:
            #         print(f'Missing Labels for {patch_name}')
            missing_count += 1
            continue

        if 'Permanently irrigated land' in patch_json['labels']:
            irrigated_examples.append(patch_folder_path)
        else:
            nonirrigated_examples.append(patch_folder_path)

    # Check for Vineyards
    vy_examples = []
    nonvy_examples = []
    missing_count = 0
    for patch_name in tqdm(patch_names_list[0]):
        patch_folder_path = os.path.join(root_folder, patch_name)
        patch_json_path = os.path.join(
            patch_folder_path, patch_name + '_labels_metadata.json')
        try:
            with open(patch_json_path, 'rb') as f:
                patch_json = json.load(f)
        except:
            #         print(f'Missing Labels for {patch_name}')
            missing_count += 1
            continue

        if 'Vineyards' in patch_json['labels']:
            vy_examples.append(patch_folder_path)
        else:
            nonvy_examples.append(patch_folder_path)

    len(vy_examples) * 2
    len(nonvy_examples)

    # New: This was added as the next code directly reads the csv and creates a dataframe
    pos_df = pd.DataFrame(irrigated_examples, columns=['file'])
    neg_df = pd.DataFrame(nonirrigated_examples, columns=['file'])
    # pos_df.to_csv('/workspace/app/data/raw/bigearthnet-models/splits/positive_train.csv')
    # neg_df.to_csv('/workspace/app/data/raw/bigearthnet-models/splits/negative_train.csv')
    pos_df.to_csv(big_earth_models_folder + 'splits/positive_val.csv')
    neg_df.to_csv(big_earth_models_folder + 'splits/negative_val.csv')

    # pos_irr_df = pd.read_csv('/workspace/app/data/raw/bigearthnet-models/splits/positive_train.csv')
    # neg_irr_df = pd.read_csv('/workspace/app/data/raw/bigearthnet-models/splits/negative_train.csv')
    pos_irr_df = pd.read_csv(big_earth_models_folder + 'splits/positive_val.csv')
    neg_irr_df = pd.read_csv(big_earth_models_folder + 'splits/negative_val.csv')

    len(pos_irr_df)

    # pos_df = pd.DataFrame(vy_examples,columns=['file'])
    # neg_df = pd.DataFrame(nonvy_examples,columns=['file'])
    # pos_df.to_csv('/workspace/app/data/raw/bigearthnet-models/splits/positive_vy_val.csv')
    # neg_df.to_csv('/workspace/app/data/raw/bigearthnet-models/splits/negative_vy_val.csv')

    # # Create Data sets for finetuning. Make total dataset size divisible by 32 or 64 for easy batching

    len(pos_df)

    pos_df_1_percent = pos_irr_df.sample(frac=0.0065)
    pos_df_3_percent = pos_irr_df.sample(frac=0.0258)
    pos_df_10_percent = pos_irr_df.sample(frac=0.103)

    print(len(pos_df_1_percent))
    print(len(pos_df_3_percent))
    print(len(pos_df_10_percent))

    sample_frac_1p = len(pos_df_1_percent) / len(neg_irr_df)
    sample_frac_3p = len(pos_df_3_percent) / len(neg_irr_df)
    sample_frac_10p = len(pos_df_10_percent) / len(neg_irr_df)

    subset_neg_df_1p = neg_irr_df.sample(frac=sample_frac_1p)
    subset_neg_df_3p = neg_irr_df.sample(frac=sample_frac_3p)
    subset_neg_df_10p = neg_irr_df.sample(frac=sample_frac_10p)

    sample_frac_ir = len(pos_df) / len(neg_df)

    neg_ir_df = neg_df.sample(frac=sample_frac_ir)

    # New
    balanced_df = pd.concat([pos_df, neg_ir_df])
    # Shuffle the examples
    balanced_df = balanced_df.sample(frac=1)
    balanced_df.to_csv(f'{big_earth_models_folder}splits/final_balanced_val.csv')

    splits = glob(f'{big_earth_models_folder}splits/final_balanced_val.*')
    patch_names_list = []
    split_names = []
    for csv_file in splits:
        patch_names_list.append([])
        split_names.append(os.path.basename(csv_file).split('.')[0])
        csv_df = pd.read_csv(csv_file)
        patch_names_list[-1] = list(csv_df.file)
        patch_names_list[-1] = [name.split('/')[-1] for name in patch_names_list[-1]]

    tensorflow_utils.prep_tf_record_files(
        root_folder, out_folder,
        split_names, patch_names_list,
        label_indices, False, True)

    pos_df = pd.DataFrame(vy_examples, columns=['file'])
    neg_df = pd.DataFrame(nonvy_examples, columns=['file'])
    pos_df.to_csv(big_earth_models_folder + 'splits/positive_val.csv')
    neg_df.to_csv(big_earth_models_folder + 'splits/negative_val.csv')

    # # Create Data sets for finetuning. Make total dataset size divisible by 32 or 64 for easy batching

    len(pos_df)

    pos_vy_df_1_percent = pos_df.sample(frac=0.0092)
    pos_vy_df_3_percent = pos_df.sample(frac=0.0366)

    print(len(pos_vy_df_1_percent))
    print(len(pos_vy_df_3_percent))

    sample_frac_vy_1p = len(pos_vy_df_1_percent) / len(neg_df)
    sample_frac_vy_3p = len(pos_vy_df_3_percent) / len(neg_df)

    subset_neg_vy_df_1p = neg_df.sample(frac=sample_frac_vy_1p)
    subset_neg_vy_df_3p = neg_df.sample(frac=sample_frac_vy_3p)

    print(len(subset_neg_vy_df_1p))
    print(len(subset_neg_vy_df_3p))

    sample_frac_vy = len(pos_df) / len(neg_df)

    neg_vy_df = neg_df.sample(frac=sample_frac_vy)

    len(neg_vy_df) * 2

    # start_index = 0
    # stop_index = 0
    # # for i in range(5):
    # #     print(f'Start Index: {start_index}')
    # #     stop_index = len(subset_neg_df)*(i+1)//5
    # #     print(f'Stop Index: {stop_index}')
    # #     balanced_df = pd.concat([pos_df, subset_neg_df[start_index:stop_index]])
    # #     start_index = stop_index
    # #     # Shuffle the examples
    # #     balanced_df = balanced_df.sample(frac=1)
    # #     balanced_df.to_csv(f'./bigearthnet-models/splits/balanced_val{i}.csv')

    balanced_df = pd.concat([pos_df, neg_vy_df])
    # Shuffle the examples
    balanced_df = balanced_df.sample(frac=1)
    balanced_df.to_csv(f'{big_earth_models_folder}splits/final_balanced_val_vy.csv')

    splits = glob(f'{big_earth_models_folder}splits/final_balanced_val_vy.*')
    patch_names_list = []
    split_names = []
    for csv_file in splits:
        patch_names_list.append([])
        split_names.append(os.path.basename(csv_file).split('.')[0])
        csv_df = pd.read_csv(csv_file)
        patch_names_list[-1] = list(csv_df.file)
        patch_names_list[-1] = [name.split('/')[-1] for name in patch_names_list[-1]]

    tensorflow_utils.prep_tf_record_files(
        root_folder, out_folder,
        split_names, patch_names_list,
        label_indices, False, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='This script creates TFRecord files for the BigEarthNet train, validation and test splits. It also shards the files.')
    parser.add_argument('-d', '--download', default=False, type=bool,
                        help="whether to download bigearthnet data")
    parser.add_argument('-tf', '--tfrecords', default=True, type=bool,
                        help="whether to create tfrecords")
    parser.add_argument('-tfl', '--tfrecordslabeled', default=True, type=bool,
                        help="whether to create tfrecords with labelled")
    args = parser.parse_args()

    if args.download:
        print('download data---START')
        download_data()
        print('download data---END')

    if args.tfrecords:
        print('preprocess_tfrecords---START')
        preprocess_tfrecords()
        print('preprocess_tfrecords---END')

    if args.tfrecordslabeled:
        print('preprocess_tfrecords_labelled---START')
        preprocess_tfrecords_labelled()
        print('preprocess_tfrecords_labelled---END')
