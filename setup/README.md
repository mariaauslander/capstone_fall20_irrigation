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

```
git clone https://github.com/Berkeley-Data/irrigation_detection.git
cd irrigation_detection
docker build -t irgapp -f ./setup/tf23.docker .

```

Run the docker 
```
docker run -u $(id -u):$(id -g) -v $PWD:/app --ipc=host --name irgapp --rm --privileged --gpus all -v /tmp:/tmp -p 8888:8888 -p 6006:6006 -ti irgapp

```

```
# once aws is configured 
-v $HOME/.aws:/root/.aws:rw
```

