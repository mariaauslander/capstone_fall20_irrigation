#!/bin/bash

#pip install --upgrade wandb
#wandb login <use your wandb apikey>
#

# ResNet50 Xception ResNet101V2

for down in 10/90 50/50
do
  for weight in False True
  do
    for architecture in InceptionV3
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
                          -u $weight \
                          -t True
        # End over percentages
        done
      # End over epochs
      done
    # End over architecture
    done
  # End over class-weight
  done
# End over class-weight
done