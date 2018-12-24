import PIL
from PIL import Image
import matplotlib.pyplot as plt
from libtiff import TIFF
from libtiff import TIFFfile, TIFFimage
from scipy.misc import imresize
import numpy as np
import glob
import cv2
import os
import math
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as keras
import tensorflow as tf
#%matplotlib inline

def resnet(x, input_shape):
    
    # Decreases the dimensions of the input image by a factor of 32
    x = ResNet50(include_top=False, weights=None, input_tensor=x, input_shape=(512,512,3)).output
    
    # Upsampling by 2
    x = UpSampling2D(size = (2,2))(x)
    ##x = BatchNormalization()(x)
    
    # Again Upsampling by 2 so that we get an output feature map of size 1/8th of the initial image
    x = UpSampling2D(size = (2,2))(x)
    ##res = BatchNormalization()(x)
    x = UpSampling2D(size = (2,2))(x)
    return x


def encoder_decoder(inp):
    
    # Encoder layers
    
    ed_conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same')(inp)
    bn1 = BatchNormalization()(ed_conv1)
    pool1 = MaxPooling2D()(bn1)
    
    ed_conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same')(pool1)
    bn2 = BatchNormalization()(ed_conv2)
    pool2 = MaxPooling2D()(bn2)
    
    # Decoder layers
    
    up1 = Conv2D(16, 3, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(pool2))
    bn3 = BatchNormalization()(up1)
    
    up2 = Conv2D(3, 3, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(bn3))
    bn4 = BatchNormalization()(up2)

    return bn4 


# Bilinear Interpolation

def interpolation(x, shape):
    
    # The height and breadth to which the pooled feature maps are to be interpolated
    h_to, w_to = shape
    
    # Bilinear Interpolation (Default method of this tf function is method=ResizeMethod.BILINEAR)
    resized = tf.image.resize_images(x, [h_to, w_to], align_corners=True)
    
    return resized


def pool_and_interp(res, level, feature_map_shape):
    
    kernel_strides_dict = {1: 30, 2: 15, 3: 10, 6: 5}
    
    # Kernels and strides according to the level
    kernel = (kernel_strides_dict[level], kernel_strides_dict[level])
    strides = (kernel_strides_dict[level], kernel_strides_dict[level])
    
    x = AveragePooling2D(kernel, strides=strides)(res)
    x = Conv2D(512, (1, 1), strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Lambda(interpolation, arguments={'shape': feature_map_shape})(x)
    return x


# Pyramid Pooling Module

def pyramid_pooling_module(res):
    
    # Output tensor shape of Resnet50(Which is 1/8th of input size)
    resnet_out_shape = K.int_shape(res)
    feature_map_shape = (resnet_out_shape[1], resnet_out_shape[2])
    
    pool_and_interp1 = pool_and_interp(res, 1, feature_map_shape)
    pool_and_interp2 = pool_and_interp(res, 2, feature_map_shape)
    pool_and_interp3 = pool_and_interp(res, 3, feature_map_shape)
    pool_and_interp6 = pool_and_interp(res, 6, feature_map_shape)
    
    # Concatenate the outputs of all the pool_and_interp module and the output feature map of ResNet
    concat = Concatenate()([res, pool_and_interp6, pool_and_interp3, pool_and_interp2, pool_and_interp1])
    
    return concat


def PSPNet(n_classes = 3, input_shape = (128, 128, 4)):
    
    # Input to the model
    inputs = Input(input_shape)
    
    '''in_shape = inputs.shape
    out_shape = (in_shape[1], in_shape[2], 3)'''
    
    # Converting 4 channel input to a 3 channel map using Encoder-Decoder network 
    # to give it as a input to ResNet50 with pretrained weights
    res_input = encoder_decoder(inputs)            
    
    res_input_shape = K.int_shape(res_input)
    res_input_shape = (res_input_shape[1],res_input_shape[2],res_input_shape[3])
    
    # Passing the 3 channel map into ResNet50 followed by 2 upsampling layers 
    # to get a output of shape exactly 1/8th of the input map shape
    res = resnet(res_input, input_shape = res_input_shape)                        
    
    # Pyramid Pooling Module
    ppmodule_out = pyramid_pooling_module(res)                
    
    # Final Conv layers and output
    x = Conv2D(512, 3, activation = 'relu', padding='same')(ppmodule_out)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Conv2D(n_classes, 1)(x)
    #x = interpolation(x, shape = (input_shape[0], input_shape[1]))
    x = Lambda(interpolation, arguments={'shape': (input_shape[0], input_shape[1])})(x)
    out = Activation('softmax')(x)
    
    model = Model(inputs = inputs, outputs = out)
    
    adam = Adam(lr = 0.00001)
    
    model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    model.summary()
    return model

model = PSPNet()
