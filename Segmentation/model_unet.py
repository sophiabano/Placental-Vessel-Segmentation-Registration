#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 15:24:52 2019

@author: sophiabano
"""

import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

from keras.layers import core



#from keras.layers import Lambda,Input, Conv2D,Conv2DTranspose,Conv2DTranspose, MaxPooling2D, UpSampling2D,Cropping2D, core, Dropout,normalization,concatenate,Activation



def unet_vanilla(pretrained_weights = None,input_size = (None,None,3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)
    
    return model
"""
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)
    #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    #if(pretrained_weights):
    #	model.load_weights(pretrained_weights)

    return model
"""

def unet_sadda(pretrained_weights = None,input_size = (None,None,3)):
    inputs = Input(input_size)

    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    #conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    #conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    #conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    #conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    #conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    #conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    #conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    #conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    #conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    #conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    #conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    #conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    #conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    #conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    #conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    #conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    #up6 = BatchNormalization()(up6)
    merge6 = concatenate([conv4,up6], axis = 3)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    #conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    #conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    #conv6 = BatchNormalization()(conv6)    
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    #conv6 = BatchNormalization()(conv6)    
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    #conv6 = BatchNormalization()(conv6)    
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    #conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    #conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    #conv6 = BatchNormalization()(conv6)
    
    up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    #up7 = BatchNormalization()(up7)
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    #conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    #conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    #conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    #conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    #conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    #conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    #conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    #conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    #conv7 = BatchNormalization()(conv7)

    conv9 = Conv2D(1, 1, activation = 'sigmoid')(conv7)
    #conv9= BatchNormalization()(conv9)


    model = Model(inputs = inputs, outputs = conv9)

    #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    #if(pretrained_weights):
    #	model.load_weights(pretrained_weights)

    return model

def unet_saddaBN(pretrained_weights = None,input_size = (None,None,3)):
    inputs = Input(input_size)

    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    #conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    #conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    #conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    #conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    #conv3 = Dropout(0.3)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    #drop4 = Dropout(0.3)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    #conv5 = Dropout(0.3)(conv5)

    up6 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    up6 = BatchNormalization()(up6)
    merge6 = concatenate([conv4,up6], axis = 3)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)    
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)    
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)    
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    #conv6 = Dropout(0.3)(conv6)
    #conv6 = BatchNormalization()(conv6)
    
    up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    up7 = BatchNormalization()(up7)
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    #conv7 = Dropout(0.3)(conv7)
    #conv7 = BatchNormalization()(conv7)

    conv9 = Conv2D(1, 1, activation = 'sigmoid')(conv7)
    conv9= BatchNormalization()(conv9)


    model = Model(inputs = inputs, outputs = conv9)

    #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    #if(pretrained_weights):
    #	model.load_weights(pretrained_weights)

    return model


def unet_att(pretrained_weights = None,input_size = (224, 224, 3)):
    #inputs = Input(input_size) # Input((self.patch_height, self.patch_width,1))
    inputs = Input(input_size)
    #_,height, width, channel = inputs.shape
    #height, width, channel = input_size
    conv1 = Conv2D(32, (3, 3), padding='same')(inputs)  # 'valid'
    conv1 = LeakyReLU(alpha=0.3)(conv1)
    conv1 = Dropout(0.2)(conv1)
    conv1 = BatchNormalization()(conv1)#epsilon=2e-05, axis=1, momentum=0.9, weights=None,beta_initializer='RandomNormal', gamma_initializer='one')(conv1)
    conv1 = Conv2D(32, (3, 3), dilation_rate=2, padding='same')(conv1)
    conv1 = LeakyReLU(alpha=0.3)(conv1)
    conv1 = Conv2D(32, (3, 3), dilation_rate=4, padding='same')(conv1)
    conv1 = LeakyReLU(alpha=0.3)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	# pool1 = normalization.BatchNormalization(epsilon=1e-06, mode=1, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(pool1)
    conv2 = Conv2D(64, (3, 3), padding='same')(pool1)  # ,activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)#epsilon=2e-05, axis=1, momentum=0.9, weights=None, beta_initializer='RandomNormal', gamma_initializer='one')(conv2)
    conv2 = LeakyReLU(alpha=0.3)(conv2)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), dilation_rate=2, padding='same')(conv2)
    conv2 = LeakyReLU(alpha=0.3)(conv2)
    conv2 = Conv2D(64, (3, 3), dilation_rate=4, padding='same')(conv2)  # ,W_regularizer=l2(0.01), b_regularizer=l2(0.01))(conv2)
    conv2 = LeakyReLU(alpha=0.3)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	# crop = Cropping2D(cropping=((int(3 * patch_height / 8), int(3 * patch_height / 8)), (int(3 * patch_width / 8), int(3 * patch_width / 8))))(conv1)
	# conv3 = concatenate([crop,pool2], axis=1)
    conv3 = Conv2D(128, (3, 3), padding='same')(pool2)  # , activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)#epsilon=2e-05, axis=1, momentum=0.9, weights=None,beta_initializer='RandomNormal', gamma_initializer='one')(conv3)
    conv3 = LeakyReLU(alpha=0.3)(conv3)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), dilation_rate=2, padding='same')(conv3)  # ,W_regularizer=l2(0.01), b_regularizer=l2(0.01))(conv3)
    conv3 = BatchNormalization()(conv3)#epsilon=2e-05, axis=1, momentum=0.9, weights=None,beta_initializer='RandomNormal', gamma_initializer='one')(conv3)
    conv3 = LeakyReLU(alpha=0.3)(conv3)
    conv3 = Conv2D(128, (3, 3), dilation_rate=4, padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)#epsilon=2e-05, axis=1, momentum=0.9, weights=None,beta_initializer='RandomNormal', gamma_initializer='one')(conv3)
    conv3 = LeakyReLU(alpha=0.3)(conv3)

	# up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=3)
    conv4 = Conv2D(64, (3, 3), padding='same')(up1)
    conv4 = LeakyReLU(alpha=0.3)(conv4)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), padding='same')(conv4)
    conv4 = LeakyReLU(alpha=0.3)(conv4)
	# conv4 = normalization.BatchNormalization(epsilon=1e-06, mode=1, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(conv4)
#
	# up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=3)
    conv5 = Conv2D(32, (3, 3), padding='same')(up2)
    conv5 = LeakyReLU(alpha=0.3)(conv5)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), padding='same')(conv5)
    conv5 = LeakyReLU(alpha=0.3)(conv5)
    conv6 = Conv2D(1, (1, 1), padding='same')(conv5)
    conv6 = LeakyReLU(alpha=0.3)(conv6)
	# conv6 = normalization.BatchNormalization(epsilon=1e-06, mode=1, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(conv6)

	# for tensorflow
    #conv6 = core.Reshape((-1,1+1))(conv6)
    #conv6 = core.Reshape((patch_height*patch_width,num_lesion_class+1))(conv6)
    # for theano
	#conv6 = core.Reshape((self.patch_height * self.patch_width,self.num_seg_class + 1))(conv6)
	#conv6 = core.Permute((2, 1))(conv6)
	############
    act = Activation('softmax')(conv6)
    
    model = Model(inputs=inputs, outputs=act)
    
    return model
        
def SDC_SB(pretrained_weights = None,input_size = (None, None, 3)):
    
    inputs = Input(input_size)   
    conv1_r1 = Conv2D(16, (5,5), dilation_rate=(1, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    ##conv1_r1 = Dropout(0.2)(conv1_r1)
    conv1_r2 = Conv2D(16, (5,5), dilation_rate=(2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    #conv1_r2 = Dropout(0.2)(conv1_r2)
    conv1_r3 = Conv2D(16, (5,5), dilation_rate=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    #conv1_r3 = Dropout(0.2)(conv1_r3)
    conv1_r4 = Conv2D(16, (5,5), dilation_rate=(4, 4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    #conv1_r4 = Dropout(0.2)(conv1_r4)
    sdc_1 = concatenate([conv1_r1, conv1_r2, conv1_r3, conv1_r4], axis=3)
    sdc_1 = Dropout(0.2)(sdc_1)

    conv2_r1 = Conv2D(32, (5,5), dilation_rate=(1, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_1)
    ##conv2_r1 = Dropout(0.2)(conv2_r1)
    conv2_r2 = Conv2D(32, (5,5), dilation_rate=(2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_1)
    #conv2_r2 = Dropout(0.2)(conv2_r2)
    conv2_r3 = Conv2D(32, (5,5), dilation_rate=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_1)
    #conv2_r3 = Dropout(0.2)(conv2_r3)
    conv2_r4 = Conv2D(32, (5,5), dilation_rate=(4, 4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_1)
    ##conv2_r4 = Dropout(0.2)(conv2_r4)
    sdc_2 = concatenate([conv2_r1, conv2_r2, conv2_r3, conv2_r4], axis=3)
    sdc_2 = Dropout(0.2)(sdc_2)
    
    conv3_r1 = Conv2D(32, (5,5), dilation_rate=(1, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_2)
    #conv3_r1 = Dropout(0.2)(conv3_r1)
    conv3_r2 = Conv2D(32, (5,5), dilation_rate=(2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_2)
    #conv3_r2 = Dropout(0.2)(conv3_r2)
    conv3_r3 = Conv2D(32, (5,5), dilation_rate=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_2)
    #conv3_r3 = Dropout(0.2)(conv3_r3)
    conv3_r4 = Conv2D(32, (5,5), dilation_rate=(4, 4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_2)
    #conv3_r4 = Dropout(0.2)(conv3_r4)
    sdc_3 = concatenate([conv3_r1, conv3_r2, conv3_r3, conv3_r4], axis=3)
    sdc_3 = Dropout(0.2)(sdc_3)
    
    conv4_r1 = Conv2D(64, (5,5), dilation_rate=(1, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_3)
    #conv4_r1 = Dropout(0.2)(conv4_r1)
    conv4_r2 = Conv2D(64, (5,5), dilation_rate=(2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_3)
    #conv4_r2 = Dropout(0.2)(conv4_r2)
    conv4_r3 = Conv2D(64, (5,5), dilation_rate=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_3)
    #conv4_r3 = Dropout(0.2)(conv4_r3)
    conv4_r4 = Conv2D(64, (5,5), dilation_rate=(4, 4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_3)
    #conv4_r4 = Dropout(0.2)(conv4_r4)
    sdc_4 = concatenate([conv4_r1, conv4_r2, conv4_r3, conv4_r4], axis=3)
    sdc_4 = Dropout(0.2)(sdc_4)

    conv5_r1 = Conv2D(32, (5,5), dilation_rate=(1, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_4)
    #conv5_r1 = Dropout(0.2)(conv5_r1)
    conv5_r2 = Conv2D(32, (5,5), dilation_rate=(2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_4)
    #conv5_r2 = Dropout(0.2)(conv5_r2)
    conv5_r3 = Conv2D(32, (5,5), dilation_rate=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_4)
    #conv5_r3 = Dropout(0.2)(conv5_r3)
    conv5_r4 = Conv2D(32, (5,5), dilation_rate=(4, 4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_4)
    #conv5_r4 = Dropout(0.2)(conv5_r4)
    sdc_5 = concatenate([conv5_r1, conv5_r2, conv5_r3, conv5_r4], axis=3)
    sdc_5 = Dropout(0.2)(sdc_5)  
    
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_5)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)
 
    
    return model
    
    
def SDC_SB6(pretrained_weights = None,input_size = (None, None, 3)):
    
    inputs = Input(input_size)   
    conv1_r1 = Conv2D(16, (5,5), dilation_rate=(1, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    ##conv1_r1 = Dropout(0.2)(conv1_r1)
    conv1_r2 = Conv2D(16, (5,5), dilation_rate=(2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    #conv1_r2 = Dropout(0.2)(conv1_r2)
    conv1_r3 = Conv2D(16, (5,5), dilation_rate=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    #conv1_r3 = Dropout(0.2)(conv1_r3)
    conv1_r4 = Conv2D(16, (5,5), dilation_rate=(4, 4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    #conv1_r4 = Dropout(0.2)(conv1_r4)
    sdc_1 = concatenate([conv1_r1, conv1_r2, conv1_r3, conv1_r4], axis=3)
    sdc_1 = Dropout(0.1)(sdc_1)

    conv2_r1 = Conv2D(32, (5,5), dilation_rate=(1, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_1)
    ##conv2_r1 = Dropout(0.2)(conv2_r1)
    conv2_r2 = Conv2D(32, (5,5), dilation_rate=(2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_1)
    #conv2_r2 = Dropout(0.2)(conv2_r2)
    conv2_r3 = Conv2D(32, (5,5), dilation_rate=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_1)
    #conv2_r3 = Dropout(0.2)(conv2_r3)
    conv2_r4 = Conv2D(32, (5,5), dilation_rate=(4, 4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_1)
    ##conv2_r4 = Dropout(0.2)(conv2_r4)
    sdc_2 = concatenate([conv2_r1, conv2_r2, conv2_r3, conv2_r4], axis=3)
    sdc_2 = Dropout(0.1)(sdc_2)
    
    conv3_r1 = Conv2D(32, (5,5), dilation_rate=(1, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_2)
    #conv3_r1 = Dropout(0.2)(conv3_r1)
    conv3_r2 = Conv2D(32, (5,5), dilation_rate=(2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_2)
    #conv3_r2 = Dropout(0.2)(conv3_r2)
    conv3_r3 = Conv2D(32, (5,5), dilation_rate=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_2)
    #conv3_r3 = Dropout(0.2)(conv3_r3)
    conv3_r4 = Conv2D(32, (5,5), dilation_rate=(4, 4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_2)
    #conv3_r4 = Dropout(0.2)(conv3_r4)
    sdc_3 = concatenate([conv3_r1, conv3_r2, conv3_r3, conv3_r4], axis=3)
    sdc_3 = Dropout(0.1)(sdc_3)
    
    conv4_r1 = Conv2D(64, (5,5), dilation_rate=(1, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_3)
    #conv4_r1 = Dropout(0.2)(conv4_r1)
    conv4_r2 = Conv2D(64, (5,5), dilation_rate=(2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_3)
    #conv4_r2 = Dropout(0.2)(conv4_r2)
    conv4_r3 = Conv2D(64, (5,5), dilation_rate=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_3)
    #conv4_r3 = Dropout(0.2)(conv4_r3)
    conv4_r4 = Conv2D(64, (5,5), dilation_rate=(4, 4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_3)
    #conv4_r4 = Dropout(0.2)(conv4_r4)
    sdc_4 = concatenate([conv4_r1, conv4_r2, conv4_r3, conv4_r4], axis=3)
    sdc_4 = Dropout(0.1)(sdc_4)

    conv5_r1 = Conv2D(32, (5,5), dilation_rate=(1, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_4)
    #conv5_r1 = Dropout(0.2)(conv5_r1)
    conv5_r2 = Conv2D(32, (5,5), dilation_rate=(2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_4)
    #conv5_r2 = Dropout(0.2)(conv5_r2)
    conv5_r3 = Conv2D(32, (5,5), dilation_rate=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_4)
    #conv5_r3 = Dropout(0.2)(conv5_r3)
    conv5_r4 = Conv2D(32, (5,5), dilation_rate=(4, 4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_4)
    #conv5_r4 = Dropout(0.2)(conv5_r4)
    sdc_5 = concatenate([conv5_r1, conv5_r2, conv5_r3, conv5_r4], axis=3)
    sdc_5 = Dropout(0.1)(sdc_5)  

    conv6_r1 = Conv2D(16, (5,5), dilation_rate=(1, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_4)
    #conv5_r1 = Dropout(0.2)(conv5_r1)
    conv6_r2 = Conv2D(16, (5,5), dilation_rate=(2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_4)
    #conv5_r2 = Dropout(0.2)(conv5_r2)
    conv6_r3 = Conv2D(16, (5,5), dilation_rate=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_4)
    #conv5_r3 = Dropout(0.2)(conv5_r3)
    conv6_r4 = Conv2D(16, (5,5), dilation_rate=(4, 4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_4)
    #conv5_r4 = Dropout(0.2)(conv5_r4)
    sdc_6 = concatenate([conv6_r1, conv6_r2, conv6_r3, conv6_r4], axis=3)
    sdc_6 = Dropout(0.1)(sdc_6)  
    
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sdc_5)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)
 
    
    return model 
        
