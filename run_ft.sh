#!/bin/bash

for epoch in 20 30 35 40 45
 do
 for perc in 1 3 10
  do
   python finetune.py -p ca_simclr_rn101_s50_t1_50_${epoch}.h5 \
                      -t final_balanced_train_${perc}percent.tfrecord \
                      -e 250 \
                      -o ca_simclr_rn101_t1_e${epoch}_mlp_${perc}perc >> ft_${epoch}_epoch.out
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