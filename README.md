# Individual Vertebrae Segmentation on VerSe Dataset

This is the implementation of iterative vertebrae segmentation on VerSe datasets (2019 and 2020). Vertebrae segmentation is accomplished by first finding the spine location on CT volume and then performing an iterative segmentation of vertebrae from top to down:
<p align="center">
  <img src="https://github.com/luiserrador/IndVertSeg_VerSe/blob/master/images/seg_algorithm.png" width=400>
</p>

Note: [ML_3D_Unet](https://github.com/luiserrador/ML_3D_Unet) needed.

# Usage

First thing to do after cloning this repo is to clone [ML_3D_Unet](https://github.com/luiserrador/ML_3D_Unet) which is needed:
```
$ git clone https://github.com/luiserrador/ML_3D_Unet.git
```
