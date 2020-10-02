# Deep Learning for Irrigation Detection
## UC Berkeley - MIDS Capstone Fall 2020
### Jade, Jay, Maria, Tom

**Note: Much of the work is contained in Colaboratory notebooks on a Shared Google Drive as initial model exploration and EDA was done collaboratively in that environment. As more organized model exploration and hyperparameter sensitivity evaluations were needed, the team focused on analysis using V100 GPUs in the IBM Cloud.**

## Training
Follow the steps below for setting up the appropriate environment for model training.

1. Requisition a GPU with > 1TB additional mounted disk. Ideally a V100 as it trains 3x faster than a P100.
2. ssh to this GPU and perform the following. and connect to cloud storage (S3Fuse was used)
3. Clone this GitHub repo
4. Run `sh ./setup/prep_workspace.sh` - this will create a directory structure expected by the model
5. Copy the necessary clouds from cloud storage to the `/root/capstone_fall20_irrigation/BigEarthData/tfrecords` directory
5. Build the docker image using the command `docker build -t irgapp -f ./setup/tf23.docker .`
6. Run the docker container interactively passing in the GitHub repo and the mounted files from cloud storage:
```docker run -it --rm -v /root/capstone_fall20_irrigation:/capstone_fall20_irrigation -v /mnt/irrigation.data:/data irgapp bash```
7. The above command will place you withini the docker container. Train the model using the following: 
```python3 supervised_classification.py -a ARCH -o OUTPUT -e EPOCHS -b BATCH```
           
           where ARCH is 'InceptionV3', 'ResNet50', 'Xception', or 'ResNet101V2'
                 OUTPUT is a prefix for model file and results file
                 EPOCHS is number of epochs to run (50 is default)
                 BATCH is batch size (default is 32)
                 
           



