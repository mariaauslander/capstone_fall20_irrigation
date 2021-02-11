

# Dataset preparation
#### BigEarthNet data splits:
- Train images:269695
- Test images:125866
- Validation images:123723
- Balanced irrigation: 6606
- Balanced vineyards: 4758

#### EDA 
# Integrating and Exploring the Combined MSI Dataset for California

#### Data Processing 
1. BigEarthNet dataset 
```
# download 66GB package 
curl http://bigearth.net/downloads/BigEarthNet-S2-v1.0.tar.gz -o data/raw/BigEarthNet-v1.0.zip

# extract 
tar -xvf data/raw/BigEarthNet-v1.0.zip -C data/raw

# download bigearthnet model
curl "https://gitlab.tubit.tu-berlin.de/rsim/BigEarthNet-S2\_43-classes\_models/repository/master/archive.zip" -o "bigearthnet-models.zip"
```
Note: The extracted files are also in S3 (s3://agcapbky/BigEarthNet/BigEarthNet-v1.0/). 
Don't download from S3 as it will take ~8 hours as there are more than 7 million files. 
Instead, use the above steps. Download from S3 only if the bigearth.net is not accessible using:
```
aws s3 cp s3://agcapbky/BigEarthNet/ ./s3_data/BigEarthNet/ --recursive
```
2. BigEarthData Into TFRecords
	1. `src/data/preprocess_tfrecords.py` 
	2. `src/data/preprocess_tfrecords_labeled.py`  (irrigation vs non-irrigation)
	3. To preprocess the GeoTiff files that contain the Sentinel-2 MSI data comprising the BigEarthNet dataset into TFRecords files. It is based on the preprocessing scripts from the BigEarthNet repo, but has been updated to work in Colaboratory with Python3.7+ and TensorFlow 2.3.
	4. input: BigEarthNet dataset
		1.  v1: s3://BigEarthNet-v1.0
		2. bigearthnet-models/label_indices.json 

Note: The generated TFRecords are also in S3 (s3://agcapbky/BigEarthNet/processed/). The data can be downloaded using 
```
aws s3 cp
```
**Note**: Instead of the above 2 steps, we can call `src/data/make_dataset.py`. It takes the following arguments:
```
-d (whether to download bigearthnet data), default is False
-tf (whether to create tfrecords), default is True
-tfl (whether to create tfrecords with labelled) default is True
```
3. California Data
California data is generated using Goole Earth Engine API. The following are the latitude and longitude ranges:  

a) Fresno to Bakersfield  
	
	lat_range = 35.125,35.375  
	lon_range = -119.875,-119.625  
	
	lat_range = 35.125,35.375  
	lon_range = -119.125,-118.875  
	
b) Sacramento to Merced  
	
	lat_range = 37.125,37.375  
	lon_range = -121.125,-120.875  
	
	lat_range = 37.125,37.375  
	lon_range = -121.125,-120.875  
	
c) Calexico Region  
	
	lat_range = 32.625,33.375  
	lon_range = -115.875,-114.875  

d) North of Sacramento  
	
	lat_range = 39.875, 40.125  
	lon_range = -122.125,-121.875  

The data is available in S3. 
- Preferred option: s3://agcapbky/CaliforniaData/raw/zip/CaliforniaData.zip  (2 GB)
- Extracted files from CaliforniaData.zip: s3://agcapbky/CaliforniaData/raw/extracted/


# Data Augmentation Pipeline Testing



#### Model Valuation 
 processing results from supervised and unsupervised models. We do this by plotting training and validation metrics by epoch for the supervised and finetuned models. We also evaluate models against our test set and look at AUC. What we are after is identifying what parameters during unsupervised training and finetuning lead to the highest overall AUC as compared to our supervised baseline which is identified in the cells below.
- `notebooks/BigEarthData/epoch_metrics notebook`
- 
