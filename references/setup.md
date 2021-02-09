## GPU instance 
#### prerequisite
- local GPU
	- docker
	- nvidia driver
	- disk space (500GB+)
- AWS
	- [p3.2xlarge (or g4dn.4xlarge)](https://towardsdatascience.com/choosing-the-right-gpu-for-deep-learning-on-aws-d69c157d8c86) instance baed on [NVIDIA Deep Learning AMI](https://aws.amazon.com/marketplace/pp/NVIDIA-NVIDIA-Deep-Learning-AMI/B076K31M1S)Note: Select 'Oregon' region as p3 instances are not available in all the regions.
	- disk space (500GB+). 
	- [TODO] other GPU instance needs to be tested and compared 

#### Setup 
using [nvidia containers](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow/tags)
```
# configure s3 access 
aws configure 

git clone https://github.com/Berkeley-Data/irrigation_detection.git

cd irrigation_detection/setup

# Build the docker image with tag 'irgapp'. If the build is successful, we should be able to see the final message 'Successfully tagged irgapp:latest'
docker build -t irgapp -f tf23.docker . --no-cache

# (optional)Display the current docker images. We should be able to see 'irgapp'
docker image ls

# PWD should on the project root directory. 
cd .. 

# Run the docker container and access its shell
docker run --name tf --gpus all -it --rm -p 8888:8888 -v $PWD:/workspace/app -v /tmp:/tmp irgapp

# run jupyter notebook inside 
jupyter notebook 
```

##### Dataset
1. [bigearthnet-s2 v1.0](http://bigearth.net/#downloads) 
	1. ![[Pasted image 20210206105642.png]]
2. [bigearthnet-model](https://gitlab.tubit.tu-berlin.de/rsim/BigEarthNet-S2_43-classes_models) 
	1. download bigearthnet-model and unzip to /data/raw/bigearthnet-models 
	```
	curl "https://gitlab.tubit.tu-berlin.de/rsim/BigEarthNet-S2_43-classes_models/repository/master/archive.zip" -o "bigearthnet-models.zip" 
	```
3. EuroSat
  * RGB
    - curl "http://madm.dfki.de/files/sentinel/EuroSAT.zip" -o "./eurosat_rgb.zip"
    - unzip eurosat_rgb.zip
  * Full
    - curl "http://madm.dfki.de/files/sentinel/EuroSATallBands.zip" -o "./eurosat_full.zip"
    - unzip eurosat_full.zip
