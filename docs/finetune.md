

pre-training with simclr 

```
simclr.py -a ResNet50 -b 128 -e 1 -t 1 -c 0

```



Example

```
python finetune.py -p ca_simclr_rn101_s50_t1_50_${epoch}.h5 \
								-t final_balanced_train_${perc}percent_SIZE \
								-e 250 \
								-o ca_simclr_rn101_t1_e${epoch}_1ph_${perc}perc \
								-u 1
```

test.py 
Download model from W&B and run evaluation

```
test_supervised.py -p bigearthnet_classification -r 3hrh6yok -c 43
```

