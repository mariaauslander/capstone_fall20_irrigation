#!/bin/bash

for epoch in 5 10 15 20 25 30 35 40 45 50
 do
 for perc in 1 3 10
  do
   python finetune.py -p ca_simclr_rn101_s50_t1_50_${epoch}.h5 \
                      -t final_balanced_train_${perc}percent_SIZE \
                      -e 250 \
                      -o ca_simclr_rn101_t1_e${epoch}_1ph_${perc}perc \
                      -u 1 >> ft_1ph_${epoch}_epoch.out
# End over percentages
done
cp ft_1ph_${epoch}_epoch.out /data
rsync -aP ./BigEarthData/models /data
# End over epochs
done
#