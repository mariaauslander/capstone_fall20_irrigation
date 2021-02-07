## GPU instance 
#### prerequisite
- local GPU
	- docker
	- nvidia driver
	- disk space (500GB+)
- AWS
	- p3.2xlarge instance baed on [NVIDIA Deep Learning AMI](https://aws.amazon.com/marketplace/pp/NVIDIA-NVIDIA-Deep-Learning-AMI/B076K31M1S)
	- disk space (500GB+)
	- [TODO] other GPU instance needs to be tested and compared 

#### Setup 
using [nvidia containers](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow/tags)
```
# configure s3 access 
aws configure 

git clone https://github.com/Berkeley-Data/irrigation_detection.git

cd irrigation_detection/setup

docker build -t irgapp -f tf22.docker . --no-cache

docker run --name tf --gpus all -it --rm -p 8888:8888 -v $HOME/.aws:/root/.aws:rw -v $PWD:/workspace/app -v /tmp:/tmp irgapp

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
