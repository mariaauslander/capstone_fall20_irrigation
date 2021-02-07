#!/usr/bin/env python
# coding: utf-8

# # Big Earth Net Preprocessing
# ## Irrigation Capstone Fall 2020
# ### TP Goter
# 
# This notebook is used to preprocess the GeoTiff files that contain the Sentinel-2 MSI data comprising the BigEarthNet dataset into TFRecords files. It is based on the preprocessing scripts from the BigEarthNet repo, but has been updated to work in Colaboratory with Python3.7+ and TensorFlow 2.3.

# In[1]:


import pandas as pd
import tensorflow as tf
from glob import glob
import os
import numpy as np
from tqdm import tqdm
import rasterio
import csv
import json


# In[2]:


print(pd.__version__)
print(tf.__version__)


# ## Mount Google Drive and Set Paths

# In[3]:


# base_path = '/content/gdrive/My Drive/Capstone Project'
big_earth_path ='/workspace/app/data/raw/BigEarthNet-v1.0/'


# ## Create Symbolic Link(s)
# Set up a symbolic link to allow for easy Python module imports. Then check to make sure the link works (it is a Unix link so check from shell)

# In[ ]:


os.system("ln -s '/workspace/app/data/raw/bigearthnet-models/' bemodels")


# In[ ]:


os.system('ls bemodels')


# ## Process All of the BigEarthNet data
# This simple script will loop over all of the subfolders in the BigEarthNet-v1.0 folder. Currently this folder does not contain the entirety of the BigEarthNet Dataset. Due to this issue, the original scripting was modified to run through the train, test, val sets and only process files if they exist. The previous script simply aborted if a file was listed in the train.csv file and was not in the directory.
# 
# ### Note: This processing takes a really long time. 
# We need to determine if there is a better way to get this data ready for ingestion into our models.

# In[4]:


from bemodels import tensorflow_utils


# In[ ]:



with open('/workspace/app/data/raw/bigearthnet-models/label_indices.json', 'rb') as f:
    label_indices = json.load(f)

root_folder = big_earth_path
out_folder = '/workspace/app/data/processed'
splits = glob(f'/workspace/app/data/raw/bigearthnet-models/splits/*.csv')

# Checks the existence of patch folders and populate the list of patch folder paths
folder_path_list = []
if not os.path.exists(root_folder):
    print('ERROR: folder', root_folder, 'does not exist')

try:
    patch_names_list = []
    split_names = []
    for csv_file in splits:
        patch_names_list.append([])
        split_names.append(os.path.basename(csv_file).split('.')[0])
        with open(csv_file, 'r') as fp:
            csv_reader = csv.reader(fp, delimiter=',')
            for row in csv_reader:
                patch_names_list[-1].append(row[0].strip())    
except:
    print('ERROR: some csv files either do not exist or have been corrupted')

tensorflow_utils.prep_tf_record_files(
    root_folder, out_folder, 
    split_names, patch_names_list, 
    label_indices, False, True)


# In[ ]:


label_indices

# [todo] not sure what the code below is for.

# raw_dataset = tf.data.TFRecordDataset("/workspace/app/data/processed/full_test.tfrecord")
#
# shards = 20
#
# for i in range(shards):
#     writer = tf.data.experimental.TFRecordWriter(f"/workspace/app/data/processed/test-part-{i}.tfrecord")
#     writer.write(raw_dataset.shard(shards, i))
#
#
# raw_dataset = tf.data.TFRecordDataset("/workspace/app/data/processed/full_train.tfrecord")
#
# shards = 50
#
# for i in range(shards):
#     writer = tf.data.experimental.TFRecordWriter(f"./tfrecords/train-part-{i}.tfrecord")
#     writer.write(raw_dataset.shard(shards, i))
#
# raw_dataset = tf.data.TFRecordDataset("/workspace/app/data/processed.tfrecord")
#
# shards = 20
#
# for i in range(shards):
#     writer = tf.data.experimental.TFRecordWriter(f"/workspace/app/data/processed/val-part-{i}.tfrecord")
#     writer.write(raw_dataset.shard(shards, i))





