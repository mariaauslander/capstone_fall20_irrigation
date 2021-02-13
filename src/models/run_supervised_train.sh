#!/bin/bash

#pip install --upgrade wandb
#wandb login <use your wandb apikey>

for epochs in 50 75 100
 do
 for percent in 1 3 5 10
  do
   python train_supervised.py -a InceptionV3 \
                      -e epochs \
                      -b 32 \
                      -p percent \
                      -t True
# End over percentages
done
# End over epochs
done