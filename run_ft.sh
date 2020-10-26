#!/bin/bash

for epoch in 18 16 14 12 10 8 6 4 2 
 do
 for perc in 1 3 13
  do
   python finetune.py -p simclr_100_t3_s50_${epoch}.h5 \
                      -t balanced_train_${perc}percent.tfrecord \
                      -e 250 \
                      -o simclr_ft_100_do50_lr0001_${perc}p_${epoch}e >> ft_${epoch}_epoch.out
# End over percentages
done
cp ft_${epoch}_epoch.out /data
# End over epochs
done
#
