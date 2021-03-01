- https://github.com/Berkeley-Data/irrigation_detection/issues/97#issuecomment-786498199

## Prepare Dataset 

- download the data from[SEN12MS website]([https://github.com/schmitt-muc/SEN12MS](https://github.com/schmitt-muc/SEN12MS)):  

- create validation split file either use the train_list.pkl file (code uses pkl) or a train_list.txt file. Will modify the code to take a sample from the train_list.pkl and write to **val_list.pkl** .

- There are a total of `162555` entries in current train_list.txt file are there. Did a random split into `88.6%:11.4%` (`test_list_updated.txt`: `val_list_updated.txt`). Chose this ratio as in dataset.py, there is a comment # `18550` samples in val set. 

- The following are the counts:
	- train_list_updated.txt/.pkl  - 144024
	- val_list_updated.txt/.pkl  - 18532

Also, the current test dataset is also around 10~11 percent of the total dataset.

Modified the classification/dataset.py (took backup as dataset_original.py) to use train_list_updated.pkl and val_list_updated.pkl. 

**main_train.py is failing with rasterio._err.CPLE_OpenFailedError:** /workspace/app/data/sen12ms/ROIs1158_spring/s2_101/ROIs1158_spring_s2_101_p138.tif : No such file or directory
The above file exists. There are no permission issues. Did a google search and found that sometimes it fails due to relative paths. It might also fail due to rasterio(gdal) issues. **Need to investigate.**

To test the above issue, I ran the following in python console inside the container and it worked fine.
```
import rasterio
with rasterio.open('/workspace/app/data/sen12ms/ROIs1158_spring/s2_101/ROIs1158_spring_s2_101_p138.tif') as src:
    print(src.width, src.height)
    print(src.crs)
    print(src.transform)
    print(src.count)
    print(src.indexes)
```
