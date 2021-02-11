#!/bin/bash

pip install --upgrade wandb
wandb login e96802b17d8e833421348df053b41a538a810177

for epochs in 60 80 100 120
 do
 for percent in 30 40
  do
   python train_supervised.py -a InceptionV3 \
                      -e epochs \
                      -b 32 \
                      -g False \
                      -p percent \
                      -t False
# End over percentages
done
# End over epochs
done