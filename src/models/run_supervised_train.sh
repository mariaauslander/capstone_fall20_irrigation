#!/bin/bash

#pip install --upgrade wandb
#wandb login <use your wandb apikey>
#

# InceptionV3

for down in 10/90 50/50
do
  for architecture in ResNet50 Xception ResNet101V2
  do
    for epochs in 100
    do
      for percent in 1 3 5 10 100
      do
        python train_supervised.py -a $architecture \
                        -e $epochs \
                        -b 64 \
                        -p $percent \
                        -d $down \
                        -u 0 \
                        -t 1
      # End over percentages
      done
    # End over epochs
    done
  # End over architecture
  done
# End over class-weight
done