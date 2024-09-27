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

print(f'Using Pandas Version: {pd.__version__}')
print(f'Using TensorFlow Version: {tf.__version__}')

# Set up a symbolic link to allow for easy Python module imports. Then check to make sure the link works (it is a Unix link so check from shell)

os.system("rm bemodels")
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

# Shards the data
# tf_main_file is a main tf file name. The function will look for corresponding train, test, and val files
# Ex: balanced, balanced_vy
def shard_tfrecords(tf_main_file):

    # Shard the Train data
    raw_dataset = tf.data.TFRecordDataset(out_folder + "/"+tf_main_file+"_train.tfrecord")
    shards = 50
    print("Sharding Train data")
    for i in range(shards):
        writer = tf.data.experimental.TFRecordWriter(f"{out_folder}/{tf_main_file}_train-part-{i}.tfrecord")
        writer.write(raw_dataset.shard(shards, i))

    # Shard the Test data
    raw_dataset = tf.data.TFRecordDataset(out_folder + "/"+tf_main_file+"_test.tfrecord")
    shards = 20
    print("Sharding Test data")
    for i in range(shards):
        writer = tf.data.experimental.TFRecordWriter(f"{out_folder}/{tf_main_file}_test-part-{i}.tfrecord")
        writer.write(raw_dataset.shard(shards, i))

    # Shard the Val data
    raw_dataset = tf.data.TFRecordDataset(out_folder + "/"+tf_main_file+"_val.tfrecord")
    shards = 20
    print("Sharding Validation data")
    for i in range(shards):
        writer = tf.data.experimental.TFRecordWriter(f"{out_folder}/{tf_main_file}_val-part-{i}.tfrecord")
        writer.write(raw_dataset.shard(shards, i))

# Count BigEarthNet posiive and negative samples
def count_bn_positive_negative():
    with open(big_earth_models_folder + 'label_indices.json', 'rb') as f:
        label_indices = json.load(f)

    root_folder = big_earth_path
    splits = glob(f'{big_earth_models_folder}splits/all.csv')

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

    print ('irrigated_examples', len(irrigated_examples))
    print('non-irrigated_examples', len(nonirrigated_examples))
    print('missing_count', missing_count)
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

    print('vineyard_examples', len(vy_examples))
    print('non-vineyard_examples', len(nonvy_examples))
    print('missing_count', missing_count)

def preprocess_tfrecords_labelled(split, ratio = '50-50', include_vineyard = False):
    with open(big_earth_models_folder + 'label_indices.json', 'rb') as f:
        label_indices = json.load(f)

    print('Called preprocess_tfrecords_labelled with split ratio ',ratio )

    root_folder = big_earth_path

    splits = glob(f'{big_earth_models_folder}splits/{split}.csv')

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
    if ratio == '50-50':
        pos_df.to_csv(big_earth_models_folder + 'splits/positive_' + split + '.csv')
        neg_df.to_csv(big_earth_models_folder + 'splits/negative_' + split + '.csv')
        # Read back
        pos_irr_df = pd.read_csv(big_earth_models_folder + 'splits/positive_' + split + '.csv')
        neg_irr_df = pd.read_csv(big_earth_models_folder + 'splits/negative_' + split + '.csv')
    elif ratio == '10-90':
        pos_df.to_csv(big_earth_models_folder + 'splits/positive_10_90_' + split + '.csv')
        neg_df.to_csv(big_earth_models_folder + 'splits/negative_10_90_' + split + '.csv')
        # Read back
        pos_irr_df = pd.read_csv(big_earth_models_folder + 'splits/positive_10_90_' + split + '.csv')
        neg_irr_df = pd.read_csv(big_earth_models_folder + 'splits/negative_10_90_' + split + '.csv')
    else: #'64', '128', '256', '512', '1024'
        pos_df.to_csv(big_earth_models_folder + 'splits/positive_'+ratio+'_'+ split + '.csv')
        neg_df.to_csv(big_earth_models_folder + 'splits/negative_'+ratio+'_'+ split + '.csv')
        # Read back
        pos_irr_df = pd.read_csv(big_earth_models_folder + 'splits/positive_'+ratio+'_'+ split + '.csv')
        neg_irr_df = pd.read_csv(big_earth_models_folder + 'splits/negative_'+ratio+'_'+ split + '.csv')


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

    neg_ir_df = None

    if ratio == '50-50':
        sample_frac_ir = len(pos_df) / len(neg_df)
        neg_ir_df = neg_df.sample(frac=sample_frac_ir)
    elif ratio == '10-90':
        # Assume current positive as 10%, take 90% negative
        neg_ir_df = neg_df.sample(n=len(pos_df) * 9)
    else: #'64', '128', '256', '512', '1024'
        pos_df = pos_df.sample(n=int(ratio))
        neg_ir_df = neg_df.sample(n=len(pos_df))

    # New
    balanced_df = pd.concat([pos_df, neg_ir_df])
    # Shuffle the examples
    balanced_df = balanced_df.sample(frac=1)
    if ratio == '50-50':
        balanced_df.to_csv(f'{big_earth_models_folder}splits/balanced_{split}.csv')
        splits = glob(f'{big_earth_models_folder}splits/balanced_{split}.csv')
    elif ratio == '10-90':
        balanced_df.to_csv(f'{big_earth_models_folder}splits/balanced_10_90_{split}.csv')
        splits = glob(f'{big_earth_models_folder}splits/balanced_10_90_{split}.csv')
    else: #'64', '128', '256', '512', '1024'
        balanced_df.to_csv(f'{big_earth_models_folder}splits/balanced_{ratio}_{split}.csv')
        splits = glob(f'{big_earth_models_folder}splits/balanced_{ratio}_{split}.csv')

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

    # Start for vineyards data
    if not include_vineyard:
        return

    pos_df = pd.DataFrame(vy_examples, columns=['file'])
    neg_df = pd.DataFrame(nonvy_examples, columns=['file'])
    if ratio == '50-50':
        pos_df.to_csv(big_earth_models_folder + 'splits/positive_vy_' + split + '.csv')
        neg_df.to_csv(big_earth_models_folder + 'splits/negative_vy_' + split + '.csv')
    elif ratio == '10-90':
        pos_df.to_csv(big_earth_models_folder + 'splits/positive_10_90_vy_' + split + '.csv')
        neg_df.to_csv(big_earth_models_folder + 'splits/negative_10_90_vy_' + split + '.csv')
    else: #'64', '128', '256', '512', '1024'
        pos_df.to_csv(big_earth_models_folder + 'splits/positive_'+ratio+'_vy_'+ split + '.csv')
        neg_df.to_csv(big_earth_models_folder + 'splits/negative_'+ratio+'_vy_'+ split + '.csv')

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

    neg_vy_df = None
    if ratio == '50-50':
        sample_frac_vy = len(pos_df) / len(neg_df)
        neg_vy_df = neg_df.sample(frac=sample_frac_vy)
    elif ratio == '10-90':
        # Assume current positive as 10%, take 90% negative
        neg_vy_df = neg_df.sample(n=len(pos_df) * 9)
    else: #'64', '128', '256', '512', '1024'
        pos_df = pos_df.sample(n=int(ratio))
        neg_vy_df = neg_df.sample(n=len(pos_df))


    balanced_df = pd.concat([pos_df, neg_vy_df])
    # Shuffle the examples
    balanced_df = balanced_df.sample(frac=1)

    if ratio == '50-50':
        balanced_df.to_csv(f'{big_earth_models_folder}splits/balanced_vy_{split}.csv')
        splits = glob(f'{big_earth_models_folder}splits/balanced_vy_{split}.csv')
    elif ratio == '10-90':
        balanced_df.to_csv(f'{big_earth_models_folder}splits/balanced_vy_10_90_{split}.csv')
        splits = glob(f'{big_earth_models_folder}splits/balanced_vy_10_90_{split}.csv')
    else: #'64', '128', '256', '512', '1024'
        balanced_df.to_csv(f'{big_earth_models_folder}splits/balanced_vy_{ratio}_{split}.csv')
        splits = glob(f'{big_earth_models_folder}splits/balanced_vy_{ratio}_{split}.csv')

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
    parser.add_argument('-tf', '--tfrecords', default=False, type=bool,
                        help="whether to create tfrecords")

    # process tf records labelled for balanced data
    parser.add_argument('-tfl', '--tfrecordslabeled', default=False, type=bool,
                        help="whether to create tfrecords with labelled")
    parser.add_argument('-s', '--split', default='train', type=str,
                        help="which dataset split to create (train,val,test)")
    parser.add_argument('-sr', '--ratio', default='50-50', choices=['50-50', '10-90', '64', '128', '256', '512', '1024'],
                        help='Split ratio')

    # Shard the data
    parser.add_argument('-sd', '--sharddata', default=False, type=bool,
                        help="whether to shard the data or not")
    parser.add_argument('-sdn', '--shardname', default='balanced', type=str,
                        help="which main file to shard")

    # Count bigearthnet positive and negative count
    parser.add_argument('-cbpn', '--countbnposneg', default=False, type=bool,
                        help="Count bigearthnet psoitive and negative")

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
        preprocess_tfrecords_labelled(args.split, args.ratio)
        print('preprocess_tfrecords_labelled---END')

    if args.sharddata:
        print('shard_tfrecords---START')
        shard_tfrecords(args.shardname)
        print('shard_tfrecords---END')

    if args.countbnposneg:
        print('count_bn_positive_negative---START')
        count_bn_positive_negative()
        print('count_bn_positive_negative---END')
