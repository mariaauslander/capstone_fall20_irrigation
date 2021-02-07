#!/usr/bin/env python
# coding: utf-8

# # Big Earth Net Preprocessing
# ## Irrigation Capstone Fall 2020
# ### TP Goter
# 
# This notebook is used to preprocess the GeoTiff files that contain the Sentinel-2 MSI data comprising the BigEarthNet dataset into TFRecords files. It is based on the preprocessing scripts from the BigEarthNet repo, but has been updated to work in Colaboratory with Python3.7+ and TensorFlow 2.3.
# 
# This version of the preprocessor is for specifically isolating the irrigated and non-irrigated examples.

# In[1]:


import pandas as pd
import tensorflow as tf
from glob import glob
import os
#from matplotlib import pyplot as plt
#%matplotlib inline
import numpy as np
from tqdm import tqdm
#from google.colab import drive
#import seaborn as sns
#from matplotlib.cm import get_cmap
#import folium
#import gdal
import rasterio
import csv
import json


# In[2]:


print(pd.__version__)
print(tf.__version__)


# ## Mount Google Drive and Set Paths

# In[3]:


#from google.colab import drive
#drive.mount('/content/gdrive')


# In[4]:


#base_path = '/content/gdrive/My Drive/Capstone Project'
big_earth_path ='./BigEarthNet-v1.0/'


# ## Create Symbolic Link(s)
# Set up a symbolic link to allow for easy Python module imports. Then check to make sure the link works (it is a Unix link so check from shell)

# In[5]:


get_ipython().system("ln -s './bigearthnet-models/' bemodels")


# In[6]:


get_ipython().system('ls bemodels')


# In[7]:


from bemodels import tensorflow_utils


# ## Process All of the BigEarthNet data
# This simple script will loop over all of the subfolders in the BigEarthNet-v1.0 folder. Currently this folder does not contain the entirety of the BigEarthNet Dataset. Due to this issue, the original scripting was modified to run through the train, test, val sets and only process files if they exist. The previous script simply aborted if a file was listed in the train.csv file and was not in the directory.
# 
# ### Note: This processing takes a really long time. 
# We need to determine if there is a better way to get this data ready for ingestion into our models.

# In[98]:


with open('./bigearthnet-models/label_indices.json', 'rb') as f:
    label_indices = json.load(f)

root_folder = big_earth_path
out_folder = './tfrecords'
splits = glob(f'./bigearthnet-models/splits/val.csv')

# Checks the existence of patch folders and populate the list of patch folder paths
folder_path_list = []
if not os.path.exists(root_folder):
    print('ERROR: folder', root_folder, 'does not exist')



# In[99]:


patch_names_list = []
split_names = []
for csv_file in splits:
    patch_names_list.append([])
    split_names.append(os.path.basename(csv_file).split('.')[0])
    with open(csv_file, 'r') as fp:
        csv_reader = csv.reader(fp, delimiter=',')
        for row in csv_reader:
            patch_names_list[-1].append(row[0].strip())    

# tensorflow_utils.prep_tf_record_files(
#     root_folder, out_folder, 
#     split_names, patch_names_list, 
#     label_indices)


# In[100]:


len(patch_names_list[0])


# In[97]:


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


# ## Check for Vineyards

# In[101]:


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


# In[111]:


len(vy_examples)*2


# In[103]:


len(nonvy_examples)


# In[17]:


pos_irr_df = pd.read_csv('./bigearthnet-models/splits/positive_train.csv')
neg_irr_df = pd.read_csv('./bigearthnet-models/splits/negative_train.csv')


# In[104]:


pos_df = pd.DataFrame(vy_examples,columns=['file'])
neg_df = pd.DataFrame(nonvy_examples,columns=['file'])
pos_df.to_csv('./bigearthnet-models/splits/positive_vy_val.csv')
neg_df.to_csv('./bigearthnet-models/splits/negative_vy_val.csv')


# # Create Data sets for finetuning. Make total dataset size divisible by 32 or 64 for easy batching

# In[96]:


len(pos_irr_df)


# In[52]:


pos_df_1_percent = pos_irr_df.sample(frac=0.0065)
pos_df_3_percent = pos_irr_df.sample(frac=0.0258)
pos_df_10_percent = pos_irr_df.sample(frac=0.103)


# In[54]:


print(len(pos_df_1_percent))
print(len(pos_df_3_percent))
print(len(pos_df_10_percent))


# In[56]:


sample_frac_1p = len(pos_df_1_percent)/len(neg_irr_df)
sample_frac_3p = len(pos_df_3_percent)/len(neg_irr_df)
sample_frac_10p = len(pos_df_10_percent)/len(neg_irr_df)


# In[58]:


subset_neg_df_1p = neg_irr_df.sample(frac=sample_frac_1p)
subset_neg_df_3p = neg_irr_df.sample(frac=sample_frac_3p)
subset_neg_df_10p = neg_irr_df.sample(frac=sample_frac_10p)


# In[60]:


print(len(subset_neg_df_1p))
print(len(subset_neg_df_3p))
print(len(subset_neg_df_10p))


# In[76]:


pos_vy_df_1_percent = pos_df.sample(frac=0.0092)
pos_vy_df_3_percent = pos_df.sample(frac=0.0366)


# In[77]:


print(len(pos_vy_df_1_percent))
print(len(pos_vy_df_3_percent))


# In[79]:


sample_frac_vy_1p = len(pos_vy_df_1_percent)/len(neg_df)
sample_frac_vy_3p = len(pos_vy_df_3_percent)/len(neg_df)


# In[81]:


subset_neg_vy_df_1p = neg_df.sample(frac=sample_frac_vy_1p)
subset_neg_vy_df_3p = neg_df.sample(frac=sample_frac_vy_3p)


# In[82]:


print(len(subset_neg_vy_df_1p))
print(len(subset_neg_vy_df_3p))


# In[105]:


sample_frac_vy= len(pos_df)/len(neg_df)


# In[106]:


neg_vy_df = neg_df.sample(frac=sample_frac_vy)


# In[110]:


len(neg_vy_df) *2


# In[27]:


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


# In[108]:


balanced_df = pd.concat([pos_df, neg_vy_df])
# Shuffle the examples
balanced_df = balanced_df.sample(frac=1)
balanced_df.to_csv(f'./bigearthnet-models/splits/final_balanced_val_vy.csv')


# In[109]:


splits = glob(f'./bigearthnet-models/splits/final_balanced_val_vy.*')
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
    label_indices)


# In[ ]:




