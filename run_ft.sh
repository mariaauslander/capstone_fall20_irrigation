#!/bin/bash

for epoch in 10
 do
 for perc in 1 3 13
  do
   python finetune.py -p simclr_100_t3_s50_${epoch}.h5 \
                      -t final_balanced_train_${perc}percent.tfrecord \
                      -e 250 \
                      -o FINAL_simclr_ft_100_do50_lr00005_${perc}p_${epoch}e >> ft_${epoch}_epoch.out
# End over percentages
done
cp ft_${epoch}_epoch.out /data
rsync -aP ./BigEarthData/models /data
# End over epochs
done
#
#python finetune.py -p simclr_100_t3_s50_${epoch}.h5 \
#                      -t balanced_train_*.tfrecord \
#                      -e 250 \
#                      -o simclr_ft_100_do50_lr00005_100p_${epoch}e >> ft_${epoch}_epoch.out
#cp ft_${epoch}_epoch.out /data
#rsync -aP ./BigEarthData/models /data