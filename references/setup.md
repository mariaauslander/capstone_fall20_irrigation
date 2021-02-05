## GPU instance 
#### prerequisite
- local GPU
	- docker
	- nvidia driver
	- disk space (500GB+)
- AWS
	- p3.2xlarge instance baed on [NVIDIA Deep Learning AMI](https://aws.amazon.com/marketplace/pp/NVIDIA-NVIDIA-Deep-Learning-AMI/B076K31M1S)
	- disk space (500GB+). Note: Select 'Oregon' region as p3 instances are not available in all the regions.
	- [TODO] other GPU instance needs to be tested and compared 

#### Setup 
using [nvidia containers](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow/tags)
```
# configure s3 access 
aws configure 

git clone https://github.com/Berkeley-Data/irrigation_detection.git

cd irrigation_detection

docker build -t irgapp -f ./setup/tf23.docker .

docker run --name tf --gpus all -it --rm -p 8888:8888 -v $HOME/.aws:/root/.aws:rw -v $PWD:/workspace/app -v /tmp:/tmp irgapp

# run jupyter notebook inside 
jupyter notebook 
```
