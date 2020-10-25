#!/bin/bash

for epoch 2 4 6 8 10 12
 do
 for perc 1 3 13
  do
   python finetune.py -p simclr_100_t3_s50_${epoch}.h5 \
                      -t balanced_train_${perc}percent.tfrecord \
                      -e 250 \
                      -o simclr_ft_100_do50_lr0001_${perc}p_${epoch}e
# End over percentages
done
# End over epochs
done
#
