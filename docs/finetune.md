

#### pre-training with simclr 

```
pip install --upgrade wandb
wandb login <your wandb apikey>
python simclr.py -a ARCH -b BATCH_SIZE -e EPOCHS -t TEMPERATURE
``` 
- *ARCH* is `InceptionV3`, `ResNet50`, `Xception`, or `ResNet101V2`
- *EPOCHS* is number of epochs to run (50 is default)
- *BATCH* is batch size (default is `32`). 
- *CLASSES*: `1` (binary classification for irrigation land) or `43` (multi-class calssficiation)  
- For example, the following command will pretrain a model with 100 epochs, 128 batch size, on full training dataset on binary classification. 
```
simclr.py -a ResNet50 -b 128 -e 100 -t 1 
```  

#### finetune 

For example
```
python finetune.py -p ca_simclr_rn101_s50_t1_50_${epoch}.h5 \
								-t final_balanced_train_${perc}percent_SIZE \
								-e 250 \
								-u 1
```

#### Evaluate using pre-trained model 
This download model from W&B and run evaluation

```
test_supervised.py -p PROJECT_ID -r RUN_ID -c 43
```
- *PROJECT_ID* is project ID 
- *RUN_ID* is run ID 
- For example, 
```
test_supervised.py -p bigearthnet_classification -r 3hrh6yok -c 43
```