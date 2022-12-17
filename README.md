# Individual Vertebrae Segmentation on VerSe Dataset

This is the implementation of iterative vertebrae segmentation on [VerSe](https://github.com/anjany/verse) datasets (2019 and 2020). Vertebrae segmentation is accomplished by first finding the spine location on CT volume and then performing an iterative segmentation of vertebrae from top to down:
<p align="center">
  <img src="https://github.com/luiserrador/IndVertSeg_VerSe/blob/master/images/seg_algorithm.png" width=500>
</p>

# 1. Usage

After moving to the repo directory, the first thing to do is to clone [ML_3D_Unet](https://github.com/luiserrador/ML_3D_Unet) repo which is needed:
```
$ git clone https://github.com/luiserrador/ML_3D_Unet.git
```
Download VerSe dataset:

<sub>**Note:** We strongly recommend to test the repository with the data available. So, test steps 1.1 and 1.2 before downloanding all the data!</sub>
```
$ python get_verse_data.py
```

## 1.1 Training spine location network
```
$ python pre_processing_heatmap.py
$ python train_heatmap.py
```

## 1.2 Training segmentation network
```
$ python pre_processing_segmentation.py
$ python train_segmentation.py
```
