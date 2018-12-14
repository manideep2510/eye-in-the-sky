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
from sklearn.metrics import confusion_matrix
from unet import UNet
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imsave
from keras import backend as keras
#%matplotlib inline

model = UNet()

# To read the images in numerical order
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


# List of file names of actual Satellite images for traininig 
filelist_trainx = sorted(glob.glob('Inter-IIT-CSRE/The-Eye-in-the-Sky-dataset/sat/*.tif'), key=numericalSort)
# List of file names of classified images for traininig 
filelist_trainy = sorted(glob.glob('Inter-IIT-CSRE/The-Eye-in-the-Sky-dataset/gt/*.tif'), key=numericalSort)

# List of file names of actual Satellite images for testing 
filelist_testx = sorted(glob.glob('Inter-IIT-CSRE/The-Eye-in-the-Sky-test-data/sat_test/*.tif'), key=numericalSort)

# Not useful, messes up with the 4 dimentions of sat images

# Resizing the image to nearest dimensions multipls of 'stride'

def resize(img, stride, n_h, n_w):
    #h,l,_ = img.shape
    ne_h = (n_h*stride) + stride
    ne_w = (n_w*stride) + stride
    
    img_resized = imresize(img, (ne_h,ne_w))
    return img_resized


# Padding at the bottem and at the left of images to be able to crop them into 128*128 images for training

def padding(img, w, h, c, crop_size, stride, n_h, n_w):
    
    w_extra = w - ((n_w-1)*stride)
    w_toadd = crop_size - w_extra
    
    h_extra = h - ((n_h-1)*stride)
    h_toadd = crop_size - h_extra
    
    img_pad = np.zeros(((h+h_toadd), (w+w_toadd), c))
    #img_pad[:h, :w,:] = img
    #img_pad = img_pad+img
    img_pad = np.pad(img, [(0, h_toadd), (0, w_toadd), (0,0)], mode='constant')
    
    return img_pad
    
    
# Adding pixels to make the image with shape in multiples of stride

def add_pixals(img, h, w, c, n_h, n_w, crop_size, stride):
    
    w_extra = w - ((n_w-1)*stride)
    w_toadd = crop_size - w_extra
    
    h_extra = h - ((n_h-1)*stride)
    h_toadd = crop_size - h_extra
    
    img_add = np.zeros(((h+h_toadd), (w+w_toadd), c))
    
    img_add[:h, :w,:] = img
    img_add[h:, :w,:] = img[:h_toadd,:, :]
    img_add[:h,w:,:] = img[:,:w_toadd,:]
    img_add[h:,w:,:] = img[h-h_toadd:h,w-w_toadd:w,:]
    
    return img_add    


# Slicing the image into crop_size*crop_size crops with a stride of crop_size/2 and makking list out of them

def crops(a, crop_size = 128):
    
    stride = 32
    
    croped_images = []
    h, w, c = a.shape
    
    n_h = int(int(h/stride))
    n_w = int(int(w/stride))
    
    # Padding using the padding function we wrote
    ##a = padding(a, w, h, c, crop_size, stride, n_h, n_w)
    
    # Resizing as required
    ##a = resize(a, stride, n_h, n_w)
    
    # Adding pixals as required
    a = add_pixals(a, h, w, c, n_h, n_w, crop_size, stride)
    
    # Slicing the image into 128*128 crops with a stride of 64
    for i in range(n_h-1):
        for j in range(n_w-1):
            crop_x = a[(i*stride):((i*stride)+crop_size), (j*stride):((j*stride)+crop_size), :]
            croped_images.append(crop_x)
    return croped_images


# Making array of all the training sat images as it is without any cropping

xtrain_list = []

for fname in filelist_trainx:
    
    # Reading the image
    tif = TIFF.open(fname)
    image = tif.read_image()
    
    crop_size = 128
    
    stride = 64
    
    h, w, c = image.shape
    
    n_h = int(int(h/stride))
    n_w = int(int(w/stride))
    
    
    image = padding(image, w, h, c, crop_size, stride, n_h, n_w)
    
    xtrain_list.append(image)
    

    # Making array of all the training gt images as it is without any cropping

ytrain_list = []

for fname in filelist_trainy:
    
    # Reading the image
    tif = TIFF.open(fname)
    image = tif.read_image()
    
    crop_size = 128
    
    stride = 64
    
    h, w, c = image.shape
    
    n_h = int(int(h/stride))
    n_w = int(int(w/stride))
    
    
    image = padding(image, w, h, c, crop_size, stride, n_h, n_w)
    
    ytrain_list.append(image)
    
    
y_train = ytrain_list
x_train = xtrain_list

del ytrain_list
del xtrain_list

xtest_list1 = []

for fname in filelist_testx:
    
    # Reading the image
    tif = TIFF.open(fname)
    image = tif.read_image()
    
    crop_size = 128
    
    stride = 32
    
    h, w, c = image.shape
    
    n_h = int(int(h/stride))
    n_w = int(int(w/stride))
    
    
    image = add_pixals(image, h, w, c, n_h, n_w, crop_size, stride)
    
    xtest_list1.append(image)


# Reading, padding, cropping and making array of all the cropped images of all the trainig sat images
trainx_list = []

for fname in filelist_trainx[:13]:
    
    # Reading the image
    tif = TIFF.open(fname)
    image = tif.read_image()
    
    # Padding as required and cropping
    crops_list = crops(image)
    #print(len(crops_list))
    trainx_list = trainx_list + crops_list
    
# Array of all the cropped Training sat Images    
trainx = np.asarray(trainx_list)


# Reading, padding, cropping and making array of all the cropped images of all the trainig gt images
trainy_list = []

for fname in filelist_trainy[:13]:
    
    # Reading the image
    tif = TIFF.open(fname)
    image = tif.read_image()
    
    # Padding as required and cropping
    crops_list =crops(image)
    
    trainy_list = trainy_list + crops_list
    
# Array of all the cropped Training gt Images    
trainy = np.asarray(trainy_list)


# Reading, padding, cropping and making array of all the cropped images of all the testing sat images
testx_list = []
 
#for fname in filelist_trainx[13]:
    
    # Reading the image
tif = TIFF.open(filelist_trainx[13])
image = tif.read_image()
    
# Padding as required and cropping
crops_list = crops(image)
    
testx_list = testx_list + crops_list
    
# Array of all the cropped Testing sat Images  
testx = np.asarray(testx_list)


# Reading, padding, cropping and making array of all the cropped images of all the testing sat images
testy_list = []

#for fname in filelist_trainx[13]:
    
# Reading the image
tif = TIFF.open(filelist_trainy[13])
image = tif.read_image()
    
# Padding as required and cropping
crops_list = crops(image)
    
testy_list = testy_list + crops_list
    
# Array of all the cropped Testing sat Images  
testy = np.asarray(testy_list)


def testing(model, trainx, trainy, testx, testy, weights_file = "model_augment.h5"):
    
    pred_train_all = []
    pred_test_all = []
    
    model.load_weights(weights_file)
    
    Y_pred_train = model.predict(trainx)
    
    for k in range(Y_pred_train.shape[0]):
    
        pred_train_all.append(Y_pred_train[k])
    
    Y_gt_train = [arr for arr in trainy]
    
    Y_pred_val = model.predict(testx)
    
    for k in range(Y_pred_val.shape[0]):
    
        pred_test_all.append(Y_pred_val[k])
    
    Y_gt_val = [arr for arr in testy]
    
    return pred_train_all, pred_test_all, Y_pred_val, Y_gt_val

#pred_train_all, pred_test_all, Y_pred_val, Y_gt_val = testing(model, trainx, trainy, testx, testy, weights_file = "model_onehot.h5")

pred_train_all, Y_gt_train, pred_val_all, Y_gt_val = testing(model, trainx, trainy, testx, testy, weights_file = "model_onehot.h5")


# Convert onehot to label
def to_class_no(y_hot_list):
    y_class_list = []
    
    n = len(y_hot_list)
    
    for i in range(n):
        
        out = np.argmax(y_hot_list[i])
        
        y_class_list.append(out)
        
    return y_class_list


def conf_matrix(Y_gt, Y_pred, num_classes = 9):
    
    total_pixels = 0
    confusion_matrix = np.zeros((num_classes, num_classes))
    
    n = len(Y_pred)
    
    for i in range(n):
        y_pred = Y_pred[i]
        y_gt = Y_gt[i]
        
        #y_pred_hotcode = hotcode(y_pred)
        #y_gt_hotcode = hotcode(y_gt)
        
        pred = np.reshape(y_pred, (y_pred.shape[0]*y_pred.shape[1], y_pred.shape[2]))
        gt = np.reshape(y_gt, (y_gt.shape[0]*y_gt.shape[1], y_gt.shape[2]))
        
        pred = [i for i in pred]
        gt = [i for i in gt]
        
        pred = to_class_no(pred)
        gt = to_class_no(gt)
        
        conf_matrix = confusion_matrix(gt, pred)
        
        pixels = len(pred)
        total_pixels = total_pixels+pixels
        
        confusion_matrix = confusion_matrix + conf_matrix
        
    final_confusion_matrix = confusion_matrix/total_pixels
    
    return final_confusion_matrix

confusion_matrix_train = conf_matrix(Y_gt_train, pred_train_all, num_classes = 9)
print(confusion_matrix_train)

confusion_matrix_test = conf_matrix(Y_gt_val, pred_val_all, num_classes = 9)
print(confusion_matrix_test)

# Convert decimal onehot encode from prediction to actual onehot code

def dec_to_onehot(pred_all):
    
    pred_all_onehot_list = []
    
    for img in pred_all:
        
        h, w, c = img.shape
        
        for i in range(h):
            for j in range(w):
                
                argmax_index = np.argmax(img[i,j])
                
                sudo_onehot_arr = np.zeros((9))
                
                sudo_onehot_arr[argmax_index] = 1
                
                onehot_encode = sudo_onehot_arr
                
                img[i,j,:] = onehot_encode
                
        pred_all_onehot_list.append[img]
        
    return pred_all_onehot_list


