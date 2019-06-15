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
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from unet import UNet
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imsave
from keras import backend as K
from iou import iou
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

for fname in filelist_trainx[:13]:
    
    # Reading the image
    tif = TIFF.open(fname)
    image = tif.read_image()
    
    crop_size = 128
    
    stride = 32
    
    h, w, c = image.shape
    
    n_h = int(int(h/stride))
    n_w = int(int(w/stride))
    
    
    image = padding(image, w, h, c, crop_size, stride, n_h, n_w)
    
    xtrain_list.append(image)
    

    # Making array of all the training gt images as it is without any cropping

ytrain_list = []

for fname in filelist_trainy[:13]:
    
    # Reading the image
    tif = TIFF.open(fname)
    image = tif.read_image()
    
    crop_size = 128
    
    stride = 32
    
    h, w, c = image.shape
    
    n_h = int(int(h/stride))
    n_w = int(int(w/stride))
    
    
    image = padding(image, w, h, c, crop_size, stride, n_h, n_w)
    
    ytrain_list.append(image)
    
    
y_train = np.asarray(ytrain_list)
x_train = np.asarray(xtrain_list)

#del ytrain_list
#del xtrain_list


# Making array of all the training sat images as it is without any cropping
    
# Reading the image
tif = TIFF.open(filelist_trainx[13])
image = tif.read_image()
    
crop_size = 128
    
stride = 32
    
h, w, c = image.shape
    
n_h = int(int(h/stride))
n_w = int(int(w/stride))
     
image = add_pixals(image, h, w, c, n_h, n_w, crop_size, stride)
    
#x_val = np.reshape(image, (1,h,w,c))
x_val = image 
   
# Making array of all the training gt images as it is without any cropping

# Reading the image
tif = TIFF.open(filelist_trainy[13])
image = tif.read_image()
    
crop_size = 128
    
stride = 32
    
h, w, c = image.shape
    
n_h = int(int(h/stride))
n_w = int(int(w/stride))
    
    
image = add_pixals(image, h, w, c, n_h, n_w, crop_size, stride)
    
#y_val1 = np.reshape(image, (1,h,w,c))
y_val = image


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


color_dict = {0: (0, 0, 0),
              1: (0, 125, 0),
              2: (150, 80, 0),
              3: (255, 255, 0),
              4: (100, 100, 100),
              5: (0, 255, 0),
              6: (0, 0, 150),
              7: (150, 150, 255),
              8: (255, 255, 255)}

def rgb_to_onehot(rgb_arr, color_dict):
    num_classes = len(color_dict)
    shape = rgb_arr.shape[:2]+(num_classes,)
    print(shape)
    arr = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(color_dict):
        arr[:,:,i] = np.all(rgb_arr.reshape( (-1,3) ) == color_dict[i], axis=1).reshape(shape[:2])
    return arr

def onehot_to_rgb(onehot, color_dict):
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in color_dict.keys():
        output[single_layer==k] = color_dict[k]
    return np.uint8(output)


def testing(model, trainx, trainy, testx, testy, weights_file = "model_oneshot.h5"):
    
    pred_train_all = []
    pred_val_all = []
    
    model.load_weights(weights_file)
    
    Y_pred_train = model.predict(trainx)
    
    for k in range(Y_pred_train.shape[0]):
    
        pred_train_all.append(Y_pred_train[k])
    
    Y_gt_train = [rgb_to_onehot(arr, color_dict) for arr in trainy]
    
    Y_pred_val = model.predict(testx)
    
    for k in range(Y_pred_val.shape[0]):
    
        pred_val_all.append(Y_pred_val[k])
    
    Y_gt_val = [rgb_to_onehot(arr, color_dict) for arr in testy]
    
    return pred_train_all, Y_gt_train, pred_val_all, Y_gt_val


def testing_diffsizes(model, trainx, trainy, testx, testy, weights_file = "model_augment.h5"):
    
    pred_train_all = []
    pred_test_all = []
    
    
    model.load_weights(weights_file)
    
    for i in range(len(trainx)):
        img = trainx[i]
        h,w,c = img.shape
        img = np.reshape(img, (1,h,w,c))
        Y_pred_train = model.predict(img)
        bb,h,w,c = Y_pred_train.shape
        Y_pred_train = np.reshape(Y_pred_train, (h,w,c))
        pred_train_all.append(Y_pred_train)
    
#    for k in range(Y_pred_train.shape[0]):
    
#        pred_train_all.append(Y_pred_train[k])
    
    Y_gt_train = [rgb_to_onehot(arr, color_dict) for arr in trainy]
    
    img = testx
    h,w,c = img.shape
    img = np.reshape(img, (1,h,w,c))
    Y_pred_test = model.predict(img)
    bb,h,w,c = Y_pred_test.shape
    Y_pred_test = np.reshape(Y_pred_test, (h,w,c))
    pred_test_all.append(Y_pred_test)
    
#    for k in range(Y_pred_val.shape[0]):
    
#        pred_test_all.append(Y_pred_val[k])
    
    Y_gt_val = [rgb_to_onehot(testy, color_dict)]
    
    return pred_train_all, Y_gt_train, pred_test_all, Y_gt_val



#pred_train_all, pred_test_all, Y_pred_val, Y_gt_val = testing(model, trainx, trainy, testx, testy, weights_file = "model_onehot.h5")

##pred_train_all, Y_gt_train, pred_val_all, Y_gt_val = testing(model, trainx, trainy, testx, testy, weights_file = "model_onehot.h5")

pred_train_13, Y_gt_train_13, pred_val_all, Y_gt_val = testing_diffsizes(model, x_train, y_train, x_val, y_val, weights_file = "model_onehot.h5")

print(pred_val_all[0].shape)
print(Y_gt_val[0].shape)
#print(len(pred_train_all))
#print(len(Y_gt_train))

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
    kappa_sum = 0
    sudo_confusion_matrix = np.zeros((num_classes, num_classes))
   
#    if len(Y_pred.shape) == 3:
#        h,w,c = Y_pred.shape
#        Y_pred = np.reshape(Y_pred, (1,))
 
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
        
#        pred.tolist()
#        gt.tolist()

        gt = np.asarray(gt, dtype = 'int32')
        pred = np.asarray(pred, dtype = 'int32')

        conf_matrix = confusion_matrix(gt, pred, labels=[0,1,2,3,4,5,6,7,8])
        
        kappa = cohen_kappa_score(gt,pred, labels=[0,1,2,3,4,5,6,7])

        pixels = len(pred)
        total_pixels = total_pixels+pixels
        
        sudo_confusion_matrix = sudo_confusion_matrix + conf_matrix
        
        kappa_sum = kappa_sum + kappa

    final_confusion_matrix = sudo_confusion_matrix
    
    final_kappa = kappa_sum/n

    return final_confusion_matrix, final_kappa

confusion_matrix_train, kappa_train = conf_matrix(Y_gt_train_13, pred_train_13, num_classes = 9)
print('Confusion Matrix for training')
print(confusion_matrix_train)
print('Kappa Coeff for training without unclassified pixels')
print(kappa_train)

confusion_matrix_test, kappa_test = conf_matrix(Y_gt_val, pred_val_all, num_classes = 9)
print('Confusion Matrix for validation')
print(confusion_matrix_test)
print('Kappa Coeff for validation without unclassified pixels')
print(kappa_test)


# Pass Confusion matrix, label to which the accuracy needs to be found, number of classes to be considered
# Returns that particular class accuracy

def acc_of_class(class_label, conf_matrix, num_classes = 8):
    
    numerator = conf_matrix[class_label, class_label]
    
    denorminator = 0
    
    for i in range(num_classes):
        denorminator = denorminator + conf_matrix[class_label, i]
        
    acc_of_class = numerator/denorminator
    
    return acc_of_class


# On training

# Find accuray of all the classes NOT considering the unclassified pixels

for i in range(8):
    acc_of_cl = acc_of_class(class_label = i, conf_matrix = confusion_matrix_train, num_classes = 8)
    print('Accuracy of class '+str(i) + ' WITHOUT unclassified pixels - Training')
    print(acc_of_cl)

# Find accuray of all the classes considering the unclassified pixels

for i in range(9):
    acc_of_cl = acc_of_class(class_label = i, conf_matrix = confusion_matrix_train, num_classes = 9)
    print('Accuracy of class '+str(i) + ' WITH unclassified pixels - Training')
    print(acc_of_cl)
    
# On validation

# Find accuray of all the classes NOT considering the unclassified pixels

for i in range(8):
    acc_of_cl = acc_of_class(class_label = i, conf_matrix = confusion_matrix_test, num_classes = 8)
    print('Accuracy of class '+str(i) + ' WITHOUT unclassified pixels - Validation')
    print(acc_of_cl)

# Find accuray of all the classes considering the unclassified pixels

for i in range(9):
    acc_of_cl = acc_of_class(class_label = i, conf_matrix = confusion_matrix_test, num_classes = 9)
    print('Accuracy of class '+str(i) + ' WITH unclassified pixels - Validation')
    print(acc_of_cl)


# Calulating over all accuracy with and without unclassified pixels

def overall_acc(conf_matrix, include_unclassified_pixels = False):
    
    if include_unclassified_pixels:
        
        numerator = 0
        for i in range(9):
        
            numerator = numerator + conf_matrix[i,i]
        
        denominator = 0   
        for i in range(9):
            for j in range(9):
                
                denominator = denominator + conf_matrix[i,j]
                
        acc = numerator/denominator
        
        return acc
    
    else:
        
        numerator = 0
        for i in range(8):
        
            numerator = numerator + conf_matrix[i,i]
        
        denominator = 0   
        for i in range(8):
            for j in range(8):
            
                denominator = denominator + conf_matrix[i,j]
                
        acc = numerator/denominator
        
        return acc


# Training

# Over all accuracy without unclassified pixels

print('Over all accuracy WITHOUT unclassified pixels - Training')
print(overall_acc(conf_matrix = confusion_matrix_train, include_unclassified_pixels = False))

# Over all accuracy with unclassified pixels

print('Over all accuracy WITH unclassified pixels - Training')
print(overall_acc(conf_matrix = confusion_matrix_train, include_unclassified_pixels = True))

# Validation

# Over all accuracy without unclassified pixels

print('Over all accuracy WITHOUT unclassified pixels - Validation')
print(overall_acc(conf_matrix = confusion_matrix_test, include_unclassified_pixels = False))

# Over all accuracy with unclassified pixels

print('Over all accuracy WITH unclassified pixels - Validation')
print(overall_acc(conf_matrix = confusion_matrix_test, include_unclassified_pixels = True))



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



color_dict = {0: (0, 0, 0),
              1: (0, 125, 0),
              2: (150, 80, 0),
              3: (255, 255, 0),
              4: (100, 100, 100),
              5: (0, 255, 0),
              6: (0, 0, 150),
              7: (150, 150, 255),
              8: (255, 255, 255)}

def rgb_to_onehot(rgb_arr, color_dict):
    num_classes = len(color_dict)
    shape = rgb_arr.shape[:2]+(num_classes,)
    print(shape)
    arr = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(color_dict):
        arr[:,:,i] = np.all(rgb_arr.reshape( (-1,3) ) == color_dict[i], axis=1).reshape(shape[:2])
    return arr

def onehot_to_rgb(onehot, color_dict):
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in color_dict.keys():
        output[single_layer==k] = color_dict[k]
    return np.uint8(output)


# Pred on train, val, test and save outputs

weights_file = "model_onehot.h5"
model.load_weights(weights_file)

#y_pred_test_all = []

xtrain_list.append(x_val)


for i_ in range(len(xtrain_list)):
    
    item = xtrain_list[i_]
    
    h,w,c = item.shape
    
    item = np.reshape(item,(1,h,w,c))
    
    y_pred_train_img = model.predict(item)
    
    ba,h,w,c = y_pred_train_img.shape
    
    y_pred_train_img = np.reshape(y_pred_train_img,(h,w,c))
    
    img = y_pred_train_img
    h, w, c = img.shape
        
    for i in range(h):
        for j in range(w):
                
            argmax_index = np.argmax(img[i,j])
                
            sudo_onehot_arr = np.zeros((9))
                
            sudo_onehot_arr[argmax_index] = 1
                
            onehot_encode = sudo_onehot_arr
                
            img[i,j,:] = onehot_encode
    
    y_pred_train_img = onehot_to_rgb(img, color_dict)

    tif = TIFF.open(filelist_trainx[i_])
    image2 = tif.read_image()
    
    h,w,c = image2.shape
    
    y_pred_train_img = y_pred_train_img[:h, :w, :]
    
    imx = Image.fromarray(y_pred_train_img)
    
    imx.save("train_predictions/pred"+str(i_+1)+".jpg")



for i_ in range(len(xtest_list1)):
    
    item = xtest_list1[i_]
    
    h,w,c = item.shape
    
    item = np.reshape(item,(1,h,w,c))
    
    y_pred_test_img = model.predict(item)
    
    ba,h,w,c = y_pred_test_img.shape
    
    y_pred_test_img = np.reshape(y_pred_test_img,(h,w,c))
    
    img = y_pred_test_img
    h, w, c = img.shape
        
    for i in range(h):
        for j in range(w):
                
            argmax_index = np.argmax(img[i,j])
                
            sudo_onehot_arr = np.zeros((9))
                
            sudo_onehot_arr[argmax_index] = 1
                
            onehot_encode = sudo_onehot_arr
                
            img[i,j,:] = onehot_encode
    
    y_pred_test_img = onehot_to_rgb(img, color_dict)

    tif = TIFF.open(filelist_testx[i_])
    image2 = tif.read_image()
    
    h,w,c = image2.shape
    
    y_pred_test_img = y_pred_test_img[:h, :w, :]
    
    imx = Image.fromarray(y_pred_test_img)
    
    imx.save("test_outputs/out"+str(i_+1)+".jpg")
