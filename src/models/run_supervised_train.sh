#!/bin/bash

#pip install --upgrade wandb
#wandb login <use your wandb apikey>

for architecture in InceptionV3 ResNet50 Xception ResNet101V2
do
  for epochs in 50 100
  do
    for percent in 1 3 5 10
    do
      python train_supervised.py -a $architecture \
                      -e $epochs \
                      -b 32 \
                      -p $percent \
                      -t True
    # End over percentages
    done
  # End over epochs
  done
# End over architecture
done