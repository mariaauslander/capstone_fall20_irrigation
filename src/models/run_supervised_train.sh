#!/bin/bash

#pip install --upgrade wandb
#wandb login <use your wandb apikey>
#

# InceptionV3
# train_supervised.py -a InceptionV3 -e 50 -b 64 -p 10 -d 50/50 -c 43 -t 1

for batch in 64 #32
do
  for weight in 0 # 1
  do
    for down in no # 50/50 10/90
    do
      for architecture in InceptionV3 ResNet50 Xception ResNet101V2
      do
        for epochs in 100
        do
          for percent in 1 3 5 10 #100
          do
            python train_supervised.py -a $architecture \
                            -e $epochs \
                            -b $batch \
                            -p $percent \
                            -d $down \
                            -u $weight \
                            -c 43 \
                            -t 1
          # End over percentages
          done
        # End over epochs
        done
      # End over architecture
      done
    # End over downsampling
    done
  # End over class-weight
  done
# End over batch
done