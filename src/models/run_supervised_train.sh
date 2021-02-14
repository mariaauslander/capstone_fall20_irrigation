#!/bin/bash

#pip install --upgrade wandb
#wandb login <use your wandb apikey>
# ResNet50 Xception ResNet101V2

for architecture in InceptionV3
do
  for epochs in 50 100
  do
    for percent in 1 3 5 10 100
    do
      python train_supervised.py -a $architecture \
                      -e $epochs \
                      -b 64 \
                      -p $percent \
                      -d 50/50 \
                      -u True \
                      -t True
    # End over percentages
    done
  # End over epochs
  done
# End over architecture
done