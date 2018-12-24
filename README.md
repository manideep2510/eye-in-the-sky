# Eye In The Sky

[Satellite Image Classification](http://inter-iit.tech/events/the-eye-in-the-sky.html), InterIIT Techmeet 2018, IIT Bombay.

## About

This repository contains the implementation of two algorithms namely [U-Net: Convolutional Networks for Biomedical
Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf) and [Pyramid Scene Parsing Network](https://arxiv.org/pdf/1612.01105.pdf) modified for the problem of satellite image classification.

## Files

- [`main_unet.py`](main_unet.py) : Python code for training the algorithm with U-Net architecture including the encoding of the ground truths.
- [`unet.py`](unet.py) : Contains our implementation of U-Net layers.
- [`test_unet.py`](test_unet.py) : Code for Testing, calculating accuracies, calculating confusion matrices for training and validation and saving predictions by the U-Net model on training, validation and testing images.
- [`Inter-IIT-CSRE`](Inter-IIT-CSRE) : Contains all the training, validation ad testing data.
- [`Comparison_Test.pdf`](Comparison_Test.pdf) : Side by side comparision of the test data with the U-Net model predictions on the data.
- [`train_predictions`](train_predictions) : U-Net Model predictions on training and validation images.
- [`plots`](plots) : Accuracy and loss plots for training and validation for U-Net architecture.
- [`Test_images`](Test_images), [`Test_outputs`](Test_outputs) : Contains test images and their predictions b the U-Net model.
- [`class_masks`](class_masks), [`compare_pred_to_gt`](compare_pred_to_gt), [`images_for_doc`](images_for_doc) : Contains several images for documentation.
- [`PSPNet`](PSPNet) : Contains training files for implementation of PSPNet algorithm to satellite image classification.

## Usage

Clone the repository, change your present working directory to the cloned directory.
Create folders with names `train_predictions` and `test_outputs` to save model predicted outputs on training and testing images (Not required now as the repo already contains these folders)

```
$ git clone https://github.com/manideep2510/eye-in-the-sky.git
$ cd eye-in-the-sky
$ mkdir train_predictions
$ mkdir test_outputs
```

For training the U-Net model and saving weights, run the below command

```
$ python3 main_unet.py
```

To test the U-Net model, calculating accuracies, calculating confusion matrices for training and validation and saving predictions by the model on training, validation and testing images.

```
$ python3 test_unet.py
```

## Now, let's discuss!
Let's now discuss 

**1. What this project is about,** 

**2. Architectures we have used and experimented with and** 

**3. Some novel training strategies we have used in the project**

### Introduction

[Remote sensing](https://www.usgs.gov/faqs/what-remote-sensing-and-what-it-used) is the science of obtaining information about objects or areas from a distance, typically from aircraft or satellites.

We realized the problem of satellite image classification as a [semantic segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) problem and built semantic segmentation algorithms in deep learning to tackle this problem.

### Data Processing during training

**The Strided Cropping:**

To have sufficient training data from the given high definition images cropping is required to train the classifier which has about 31M parameters of our U-Net implementation. The crop size of 64x64 we find under-representation of the individual classes and the geometry and continuity of the objects is lost, decreasing the field of view of the convolutions.
Using a cropping window of 128x128 pixels with a stride of 32 resultant of 15887 training 414 validation images.
