# Deep Learning for Irrigation Detection
## UC Berkeley - MIDS Capstone Fall 2020
### Jade, Jay, Maria, Tom
### Website: https://people.ischool.berkeley.edu/~mariaauslander/index.html

## Overview
The intent of our work is to develop a deep neural network that will be pre-trained on Sentinel-2 Multi-spectral Satellite Imagery (MSI) from the agricultural regions of California that can be finetuned with limited data to accurately predict irrigated lands. We will use unsupervised techniques inspired by [SimCLR](https://arxiv.org/pdf/2002.05709.pdf) and [SimCLRv2](https://arxiv.org/pdf/2006.10029.pdf) from Google. These techniques have been developed and demonstrated on normal 3-channel (i.e., RGB data) for ImageNet style imagery. The clear differences between MSI (10+ channels) and ImageNet images introduce some challenges in extrapolating the SimCLR techniques to MSI data. These challenges are predominantly focused around identifying effective augmentation techniques that can be implemented as part of the contrastive learning methodology employed by the SimCLR technique. In the original [SimCLR paper](https://arxiv.org/pdf/2002.05709.pdf) numerous evaluations of different augmentation techniques and magnitudes of augmentations were performed in order to determine the best subset of augmentations to use in order to maximize model accuracy with minimal labeled data. In order to do this they evaluated a range of augmentation techniques individually and evaluated the Top-1 Accuracy on ImageNet. The results are shown in the Figure below. There were two main conclusions from this study:  1. No single augmentation technique was sufficient to achieve the accuracy they were after (i.e., > 70%) and  2. color distortion appeared the most effective augmentation technique studied.

![SimCLR Data Augmentation Evaluations](images/da_sensitivity.png)

These studies took significant compute resources which are unavailable to our team, so we start our evaluations by evaluting only the data augmentation techniques determined to be most important based on the previous work on ImageNet while acknowledging differences in our imagery data. We look at geometric modifications such as rotations, flips, shifts and zooms, color distortion (some of the techniques are only applied to the RGB channels, others on all channels), and Gaussian blurring.

Following on the work discussed in the [SimCLRv2 paper](https://arxiv.org/pdf/2006.10029.pdf), we consider that a deep, narrow network such as ResNet101 may lead to improved performance over shorter, wider networks. This is where our SimCLR evaluations began. As noted above we apply geometric, color and blur augmentations with random chances of 100%, 80% (20% for color dropping) and 50%, respectively - consistent with the SimCLR literature. We then perform sensitivities about these baseline parameters to determine whether more optimized augmentation hyperparameters exist for our MSI data. In our initial training, loss was not significantly reduced implying learning was minimal during training. We hypothesized this might be due to too strong of color jitter being applied. When reducing the color jitter intensity factor by 50%, we could see marked reductions in the contrastive loss during unsupervised training. It is expected that reducing augmentation intensity will naturally reduce the contrastive loss as the postive pairs (i.e., two augmented version of the same image) will be more similar to one another. Thus, finetuning of our model and evaluating against the validation and test sets is also needed to better distinguish between unsupervised models. 

Results from runs of our SimCLR model using 100% of the BigEarthNet training data and then finetuning on 1%, 3% and 10% of the positive class (and equal amount of negative class) images are shown in the figure below (left-hand side). The figure clearly shows a significant benefit from the SimCLR pretraining when training when using less than 10% of the positive class training examples. In fact with only 64 total images in finetuning, we see a marked increase in F1-score of ~0.4 absolute relative to a supervised baseline model. These results indicate that this technique may greatly reduce the need for labeled MSI data while also producing reasonably accurate results. However, results are still not as good as the supervised baseline. 

Separately we have also developed an independent dataset of Sentinel-2 satellite imagery focused on the agricultural regions of the state of California (see notebook for visualization of this data). We pretrain a SimCLR model using this other dataset, hoping to learn extensible features from our satellite imagery. We then finetune this California based SimCLR model using the BigEarthNet data and still see significant benefits from this non-geographically specific pretraining. This is also an encouraging result suggesting that perhaps region-specific data sets are not required for getting significant value from pretraining. These results are shown in the lower right-hand figure.

![BigEarthNet SimCLR Comparison](images/f1score_ca.png.png)

## Supervised Baseline Training - BigEarthNet Data
1. Follow the setup instructions in the Readme in the Setup Folder to install the docker container.
2. Run the docker container interactively passing in the mounted files from cloud storage:  
`nvidia-docker run -it --rm -v /mnt/irrigation_data:/data irgapp bash`
3. From within the docker container, copy the necessary clouds from cloud storage to the `/root/capstone_fall20_irrigation/BigEarthData/tfrecords` directory
4. The #6 command command will place you within the docker container. Train the model using the following:  
`python3 supervised_classification.py -a ARCH -o OUTPUT -e EPOCHS -b BATCH -g AUGMENT`
 where ARCH is 'InceptionV3', 'ResNet50', 'Xception', or 'ResNet101V2'
                 OUTPUT is a prefix for model file and results file
                 EPOCHS is number of epochs to run (50 is default)
                 BATCH is batch size (default is 32). 
                 AUGMENT is True or False (whether to use data augmentation).
                 
## Notes on Data Augmentation
Data augmentation is tested on our supervised model to ensure that:
1. the pipeline works (making it easier to implement in our unsupervised model)
2. gather insight into the effectiveness of different techniques with msi data

What we are looking for is data augmentation techniques that at the very least 'do no harm' to our supervised baseline. If we implement data augmentation techniques that make our model perform worse, it would be indicative of the fact that we are destroying important information in our inputs. We expect that techniques that work for supervised learning should also work for unsupervised learning, but the magnitude of the augmentation may need to be tuned as discussed in the [SimCLR paper](https://arxiv.org/pdf/2002.05709.pdf).

## SimCLR Model Training - BigEarthNet Data
1. Follow the setup instructions in the Readme in the Setup Folder to install the docker container.
2. Run the docker container interactively passing in the mounted files from cloud storage:  
`nvidia-docker run -it --rm -v /mnt/irrigation_data:/data irgapp bash`
3. From within the docker container, copy the necessary clouds from cloud storage to the `/root/capstone_fall20_irrigation/BigEarthData/tfrecords` directory
4. The #6 command command will place you within the docker container. Train the model using the following:  
`nohup python simclr.py -a ARCH -o OUTPUT -e EPOCHS -b BATCH -t TEMPERATURE -c CALIFORNIA&`
 where ARCH is 'InceptionV3', 'ResNet50', 'Xception', or 'ResNet101V2'
                 OUTPUT is a prefix for model file and results file
                 EPOCHS is number of epochs to run (50 is default)
                 BATCH is batch size (default is 32). 
                 TEMPERATURE is the temperature used in the contrastive loss function (default 0.1)
                 CALIFORNIA is True or False and specifies whether to train on California data (True) or BigEarthNet data (False).
 5. The above command will run the python script in background which will allow for grabbing model files as soon as they are saved. You can watch the progress by using the command below.
 ```
 tail -f nohup.out
 ```

## Fine-tuning of Pre-trained Model
1. Finetuning is done with small subsets of the actual training data. To date two subsets have been created - one with 13% of positive class images and one with ~2.7% of positive class images. These break points were chosen as they are compatible with batch sizes of 32. The 13% balanced training set (i.e. equal number of positive and negative example) contains 640 total image (320 positive and negative). The 3% balanced training set contains only 128 total images (64 of each class).
2. Due to the small training set sizes, the finetuning has been performed locally, but could easily be performed using the same docker container above.
3. During the finetuning, the SimCLR neural encoder and first two layers of the projection head are used with all weights frozen.
4. Two additional MLP layers were added and trained during the finetuning. The size of these layers is nominally set to 256, but we treat these sizes as a hyperparameter.
5. Initial studies were performed with 25% dropout following each MLP layer and a default Adam Optimizer learning rate of 1e-3.
6. Following initial trials it was observed that the learning rate should be reduced to provide for more stable learning (as expected). We currently finetune with a learning rate of 1e-4 and 50% dropout.
7. You can use the following command to run the finetuning:
```
python finetune.py -p PRETRAIN -t TRAINING_SET -o OUTPUT -e EPOCHS -b BATCH_SIZE -c CLASS
```
where PRETRAIN is the filename of the pretrained simclr model (should reside in ./BigEarthData/models directory)
      TRAINING_SET is either balanced_train_3percent.tfrecord or balanced_train_13percent.tfrecord (files should be in ./BigEarthData/tfrecords directory)
      OUTPUT is the output filename prefix
      EPOCHS is the number of epochs to finetune for - 10 is the default but we find that with the 1e-4 learning rate, 20-30 may be better. Early stopping is employed with patience of 15 epochs
      BATCH_SIZE is the batch size to use during training with a default of 32.
      CLASS is either Irrigation or Vineyards and specifies which class to finetune on
8. The above script will train and store a keras model file and a pickled data frame with epoch-wise training and validation metrics that can be plotted with the epoch_metrics notebook in the ./notebooks directory

                 
           



