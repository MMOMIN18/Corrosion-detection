#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 13:40:23 2019

@author: mobinmomin
"""


import keras
import tensorflow as tf
from keras.applications import VGG16
import os, shutil
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np
import skimage
import cv2
from skimage import feature
from skimage import morphology
from time import time
print(keras.__version__)
print(tf.__version__)

conv_base = VGG16(weights='imagenet',include_top=False,input_shape=(150, 150, 3))
conv_base.summary()

def create_directory():
    
    dataset = '/Users/mobinmomin/Desktop/Project2019/CorrosionDetector-master/download'
    dataset_rust = os.path.join(dataset, 'rust')
    dataset_norust = os.path.join(dataset, 'norust')

    directory_base = '/Users/mobinmomin/Desktop/Project2019/CorrosionDetector-master/rustnorust_b'


    directory_train = os.path.join(directory_base, 'train')
    directory_validation = os.path.join(directory_base, 'validation')
    directory_test = os.path.join(directory_base, 'test')   

    rust_directory_training = os.path.join(directory_train, 'rust')

    norust_directory_training = os.path.join(directory_train, 'norust')

    rust_directory_validation = os.path.join(directory_validation, 'rust')

    norust_directory_validation = os.path.join(directory_validation, 'norust')

    rust_directory_testing = os.path.join(directory_test, 'rust')

    norust_directory_testing = os.path.join(directory_test, 'norust')

    os.mkdir(directory_base)
    os.mkdir(directory_train)
    os.mkdir(directory_validation)
    os.mkdir(directory_test)
    os.mkdir(rust_directory_training)
    os.mkdir(norust_directory_training)
    os.mkdir(rust_directory_validation)
    os.mkdir(norust_directory_validation)
    os.mkdir(rust_directory_testing)
    os.mkdir(norust_directory_testing)

    fnames = ['rust.{}.jpg'.format(i) for i in range(70)]
    for fname in fnames:
        src = os.path.join(dataset_rust, fname)
        dst = os.path.join(rust_directory_training, fname)
        shutil.copyfile(src, dst)
    
    fnames = ['rust.{}.jpg'.format(i) for i in range(70, 76)]
    for fname in fnames:
        src = os.path.join(dataset_rust, fname)
        dst = os.path.join(rust_directory_validation, fname)
        shutil.copyfile(src, dst)
    
    fnames = ['rust.{}.jpg'.format(i) for i in range(76, 82)]
    for fname in fnames:
        src = os.path.join(dataset_rust, fname)
        dst = os.path.join(rust_directory_testing, fname)
        shutil.copyfile(src, dst)
    
    fnames = ['norust.{}.jpg'.format(i) for i in range(60)]
    for fname in fnames:
        src = os.path.join(dataset_norust, fname)
        dst = os.path.join(norust_directory_training, fname)
        shutil.copyfile(src, dst)
    
    fnames = ['norust.{}.jpg'.format(i) for i in range(60, 66)]
    for fname in fnames:
        src = os.path.join(dataset_norust, fname)
        dst = os.path.join(norust_directory_validation, fname)
        shutil.copyfile(src, dst)
    
    fnames = ['norust.{}.jpg'.format(i) for i in range(63, 72)]
    for fname in fnames:
        src = os.path.join(dataset_norust, fname)
        dst = os.path.join(norust_directory_testing, fname)
        shutil.copyfile(src, dst)
    
    print('total training rust images:', len(os.listdir(rust_directory_training)))
    print('total training norust images:', len(os.listdir(norust_directory_training)))
    print('total validation rust images:', len(os.listdir(rust_directory_validation)))
    print('total validation norust images:', len(os.listdir(norust_directory_validation)))
    print('total test rust images:', len(os.listdir(rust_directory_testing)))
    print('total test norust images:', len(os.listdir(norust_directory_testing)))

    return

def models_assignment():
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    print('This is the number of trainable weights '
      'before freezing the conv base:', len(model.trainable_weights))

    conv_base.trainable = False
    print('This is the number of trainable weights '
      'after freezing the conv base:', len(model.trainable_weights))

    return model

def training():
    
    directory_base = '/Users/mobinmomin/Desktop/Project2019/CorrosionDetector-master/rustnorust_b'
    directory_train = os.path.join(directory_base, 'train')
    directory_validation = os.path.join(directory_base, 'validation')

    data_generation_training = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

    data_generation_testing = ImageDataGenerator(rescale=1./255)

    generator_training = data_generation_training.flow_from_directory(
            directory_train,
            target_size=(150, 150),
            batch_size=4,
            class_mode='binary')


    generator_validator = data_generation_testing.flow_from_directory(
            directory_validation,
            target_size=(150, 150),
            batch_size=16,
            class_mode='binary')
    
    return generator_training, generator_validator

def tesnor_assignment():
    model = models_assignment()
    generator_training, generator_validator = training()
    tensorboard = keras.callbacks.TensorBoard(log_dir='output/{}'.format(time()))

    model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=2e-5),metrics=['acc'])

    plotting = model.fit_generator(generator_training,steps_per_epoch=10,epochs=15,validation_data=generator_validator,validation_steps=20,verbose=2,callbacks=[tensorboard])

    model.save('/Users/mobinmomin/Desktop/Project2019/CorrosionDetector-master/rustnorust_b/rustnorust_model.h5') 

    acc = plotting.history['acc']
    val_acc = plotting.history['val_acc']
    loss = plotting.history['loss']
    val_loss = plotting.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')

    plt.legend()

    plt.show()
    
    return model
  
def rust_norust(model, img_path):

    img = image.load_img(img_path, target_size=(150, 150))

    plt.imshow(img)
    test_x = image.img_to_array(img)
    test_x = test_x.reshape((1,) + test_x.shape)
    test_x = test_x.astype('float32') / 255
    rust_prob = model.predict(test_x)
    if (rust_prob > 0.50):
        print("This is a Rust image")
        return True
    else:
        print("This is a no Rust image")
    
    return False

np.set_printoptions(precision=3,suppress=True)

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.titlesize'] = 8

def image_loader(img_path):
	return cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)


def img_set_dimension(img, maximum_size=1000000):
	img1 = image_loader(img)
	return scaleDown(img1,maximum_size)

def scaleDown(img,maximum_size):
	size_of_image = img[:,:,0].size
		
	if maximum_size != None and size_of_image > maximum_size:   
		strip = np.sqrt(maximum_size/float(size_of_image))
		re_strip = re_stripping(img,strip)
		return padding(re_strip)
	
	return padding(img)

def re_stripping(img,strip):
	stripping = skimage.transform.rescale(img,strip,mode='reflect',preserve_range=True)
	return stripping.astype(np.ubyte)

def padding(arr,req=(8,8)):
	curve = np.asarray(arr.shape[0:2])
	req = np.asarray(req)
	over = np.mod(curve,req)
	adding = np.mod(req-over,req)
	broad = ((0,adding[0]),(0,adding[1]),(0,0))
	return skimage.util.pad(arr,tuple(broad),'edge')

def get_Grey_Level_Co_Matrix(grey,dist=1,ang=0):
    grey = cv2.cvtColor(grey, cv2.COLOR_RGB2GRAY)    
    return feature.greycomatrix(grey,[dist],[ang],symmetric=True,normed=True)

def Grey_Level_Co_Matrix_Energy(gray,dist=5,ang=0):
	grey_level_c_m = get_Grey_Level_Co_Matrix(gray,dist,ang)
	energy = feature.greycoprops(grey_level_c_m,prop="energy")
	return energy[0][0]

def get_Structure(bin):
	center = (bin[:-1] + bin[1:]) / 2
	width = 1.0*(bin[1] - bin[0])
	return center,width

def drawCodedBlockColor(image,index,d,value,column):
	a,b = index
	rrrr,cccc = skimage.draw.polygon(np.array([a,a,a+d,a+d]),np.array([b,b+d,b+d,b]))
	skimage.draw.set_color(image,(rrrr,cccc),column,alpha=value)
	return image

def drawCodedBlocksColor(image,index,d,value):
	plotted = np.copy(image)
	col = np.array([128,255,0])
	for i in range(len(index)):
		plotted = drawCodedBlockColor(plotted,index[i],d,value[i],col)
	
	return plotted


def hist_plot(data,nbins=256,range=None):
	his,bins = np.histogram(data,bins=nbins,density=True,range=range)
	center,width = get_Structure(bins)
	return his,center,width

def draw_image(img1,img2,titles=[],colorbar=False):
	fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(10,4))
	im1 = ax[0].imshow(img1)
	im2 = ax[1].imshow(img2)
	ax[0].set_xticks([]); ax[0].set_yticks([]); ax[1].set_xticks([]); ax[1].set_yticks([]);
	
	if len(titles) != 0:
		ax[0].set_title(titles[0])
		ax[1].set_title(titles[1])
	if colorbar:
		fig.colorbar(im1,ax=ax[0],fraction=0.046,pad=0.04)
		fig.colorbar(im2,ax=ax[1],fraction=0.046,pad=0.04)

def Energies_of_GLCM(img,d,thr,dist,thrE):
    redBlocks,redInd = get_Blocks_of_red(img,get_colour_Red(img),d,thr)
    energies = []; keepBlocks = []; keepInd = []
    
    for i in range(len(redBlocks)):
        if redBlocks[i].shape[0] == redBlocks[i].shape[1]:
            en = Grey_Level_Co_Matrix_Energy(redBlocks[i],dist=dist)
            if en < thrE:
                energies.append(en)
                keepBlocks.append(redBlocks[i]); keepInd.append(redInd[i])
            
    print(len(keepBlocks))
    return np.asarray(energies),keepBlocks,keepInd

def Hist_GLCM(img,d,thr,dist,thrE):
    energies,redBlocks,redInd = Energies_of_GLCM(img,d,thr,dist,thrE)
    
    if energies.size == 0:
        return energies,redBlocks,redInd
    
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,3))
    his,center,wid = hist_plot(energies,nbins=100)
    ax.bar(center,his,width=wid)
    ax.set_xlim([0,1.2*np.amax(energies)])
    
    return energies,redBlocks,redInd

def Hist_Energy_of_GLCM(imgs,d,thr,dist,thrE):

    energies,redBlocks,redInd = Hist_GLCM(imgs,d,thr,dist,thrE)
    masked = getMasked(imgs,get_colour_Red(imgs))
    if energies.size > 0:
        plotted = drawCodedBlocksColor(masked,redInd,d,0.5*energies / np.amax(energies))
    else:
        plotted = masked
    draw_image(imgs,plotted)

    return



def convert_img_toHSV(rgb,form="sk"):
	if form == "sk":
		return skimage.util.img_as_float(cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV))
	elif form == "cv2":
		return cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)

def look_block(image,index,d):
	x,y = index
	if len(image.shape) == 2:
		return image[x:x+d,y:y+d]
	else:
		return image[x:x+d,y:y+d,:]
    
def search_blocks(img,d):
	blocks = []; index = []
	for i in range(0,img.shape[0],d):
		for j in range(0,img.shape[1],d):
			block = look_block(img,(i,j),d)
			blocks.append(block)
			index.append((i,j))
	return blocks,index

def mask_grey(mask):
	if mask.ndim == 2:
		return mask
	return mask[:,:,0]

def mask_coloured(mask):
	if mask.ndim == 3:
		return mask
	return np.stack([mask,mask,mask],axis=-1).astype(bool)

def getMasked(image,masking1):    
	masking = np.copy(image)
	
	if image.ndim == 3 and masking1.ndim == 2:
		masking1 = mask_coloured(masking1)
	elif image.ndim == 2 and masking1.ndim == 3:
		masking1 = mask_grey(masking1)
	masking[np.logical_not(masking1)] = 0
	return masking

def get_Blocked_Mask(masking,index,d):
	i,j = index
	return masking[i:i+d,j:j+d,:]


def labelling(mask):
	return skimage.measure.label(mask.astype(int),connectivity=1,return_num=True)


def Range_of_data(image):
	return skimage.util.dtype_limits(image)

def hist2d(img,mask,nbins=256,comps=(0,1)):
	comp1 = img[:,:,comps[0]]
	comp2 = img[:,:,comps[1]]
	x = comp1[mask_grey(mask)]
	y = comp2[mask_grey(mask)]
	
	return np.histogram2d(x,y,bins=nbins,range=[[0,1],[0,1]],normed=True)


def Range_of_red():


	lower = np.array([2,60,75]) # lower H,S,V
	upper = np.array([12,250,250]) # upper H,S,V

	return lower,upper


def Mask_of_red(img):
	hsv = convert_img_toHSV(img,form="cv2")
	lower,upper = Range_of_red()
	
	return cv2.inRange(hsv,lower,upper)


def get_Blocks_of_red(image,mask,d,Thresh_block,useMasked=False):
	Blocks = []; Index = []
	
	if useMasked:
		blocks,index = search_blocks(getMasked(image,mask),d)
	else:
		blocks,index = search_blocks(image,d)
	
	for i in range(len(blocks)):
		bmask = get_Blocked_Mask(mask,index[i],d)
		prop = np.sum(bmask[:,:,0]) / d**2
		
		if prop > Thresh_block and bmask.shape[:2] == (d,d):
			Blocks.append(blocks[i])
			Index.append(index[i])
	
	return Blocks,Index


def Mask_clear(mask,minimum_Pixel,radius=4):
	mask = morphology.remove_small_objects(labelling(mask)[0],min_size=minimum_Pixel)
	mask = morphology.remove_small_holes(labelling(mask)[0],min_size=minimum_Pixel) 
	
	selem = morphology.disk(radius)
	masking = morphology.binary_closing(mask,selem=selem)

	return masking


def get_colour_Red(image,minimum_Pixel=64,clean=True):
	mask = Mask_of_red(image)
	if clean: mask = Mask_clear(mask,minimum_Pixel)
	
	return np.stack([mask,mask,mask],axis=-1).astype(bool)



def get_corroded_region(rust_norust, img_path):
    if rust_norust:
        d = 15
        thr = 0.8
        dist = 5
        thrE = 0.07
    
        img = img_set_dimension(img_path, maximum_size=1000000)
        
        Hist_Energy_of_GLCM(img,d,thr,dist,thrE)
        plt.tight_layout()
        plt.show()
        
    else:
         print("This is a no rust Image:")
    
    return
  
create_directory()
model = tesnor_assignment()
img_path_rust_norust ='/Users/mobinmomin/Desktop/Project2019/CorrosionDetector-master/rustnorust_b/train/rust/rust.10.jpg'
rust_norust1 = rust_norust(model, img_path_rust_norust)
get_corroded_region(rust_norust1, img_path_rust_norust)
