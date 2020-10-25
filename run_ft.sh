#!/bin/bash

foreach epoch ( 2 4 6 8 10 )
 foreach perc (1 3 13)
  python finetune.py -p simclr_100_t3_s50_${epoch}.h5 \
                     -t balanced_train_${perc}percent.tfrecord \
                     -e 250 \
                     -o simclr_ft_100_do50_lr00005_${perc}p_${epoch}e
# End over percentages
end
# End over epochs
end
#
