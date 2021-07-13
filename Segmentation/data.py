"""
@author: Sophia Bano

This is the implmentation of the paper 'Deep placental Vessel Segmentation for 
Fetoscopic Mosaicing' which was presented at MICCAI2020.

Note: If you use this code, consider citing the following paper: 
    
Bano, S., Vasconcelos, F., Shepherd, L.M., Vander Poorten, E., Vercauteren, T., 
Ourselin, S., David, A.L., Deprest, J. and Stoyanov, D., 2020, October. 
Deep placental vessel segmentation for fetoscopic mosaicking. In International 
Conference on Medical Image Computing and Computer-Assisted Intervention 
(pp. 763-773). Springer, Cham.

"""

from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import random
import shutil


def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        if(len(mask.shape) == 4):
            mask = np.mean(mask,axis=3)
            
        mask = (mask >0).astype(np.int)
        mask = np.expand_dims(mask,axis =3)
    return (img,mask)

def trainGenerator(batch_size,train_path,image_folder,mask_folder,preprocess_input, aug_dict,image_color_mode = "rgb",
                    mask_color_mode = "rgb",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False ,num_class = 1,save_to_dir = "./results/vis/",target_size = (448,448),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed,
        interpolation='bilinear')
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed,
        interpolation='nearest')
    
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        crop_length = 224
        batch_cropsX = np.zeros((img.shape[0], crop_length, crop_length, 3))
        batch_cropsY = np.zeros((mask.shape[0], crop_length, crop_length, 3))
        
        for i in range(img.shape[0]):
            batch_cropsX[i], batch_cropsY[i] = random_crop(img[i,:,:,:], mask[i,:,:,:], (crop_length, crop_length))
                
        batch_cropsX,batch_cropsY = adjustData(batch_cropsX,batch_cropsY,flag_multi_class,num_class)

        yield (batch_cropsX,batch_cropsY)


def valGenerator(batch_size,val_path,image_folder,mask_folder,preprocess_input, aug_dict,image_color_mode = "rgb",
                    mask_color_mode = "rgb",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False ,num_class = 1,save_to_dir = "./results/vis/",target_size = (448,448),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        val_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        val_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    
    val_generator = zip(image_generator, mask_generator)
    for (img,mask) in val_generator:
        crop_length = 224
        batch_cropsX = np.zeros((img.shape[0], crop_length, crop_length, 3))
        batch_cropsY = np.zeros((mask.shape[0], crop_length, crop_length, 3))
        
        for i in range(img.shape[0]):
            batch_cropsX[i], batch_cropsY[i] = random_crop(img[i,:,:,:], mask[i,:,:,:], (crop_length, crop_length))
                
        batch_cropsX,batch_cropsY = adjustData(batch_cropsX,batch_cropsY,flag_multi_class,num_class)

        yield (batch_cropsX,batch_cropsY)


def testGenerator(test_path,preprocess_input,num_image = 30,target_size = (448,448),flag_multi_class = False,as_gray = False):
    test_input_names =[]
    for file in np.sort(os.listdir(test_path)):
        cwd = os.getcwd()
        test_input_names.append(cwd + "/" + test_path +'/'+ file)
            
    for i in range(len(test_input_names)):
        img = io.imread(test_input_names[i],as_gray = as_gray)
        img = img / 255
        img = np.reshape(img,(1,)+img.shape)
        img = preprocess_input(img)
        yield img


def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr

def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255

def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)
        
def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)
                
def random_crop(imgX, imgY, random_crop_size):
    assert imgX.shape[2] == 3
    height, width = imgX.shape[0], imgX.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return (imgX[y:(y+dy), x:(x+dx), :], imgY[y:(y+dy), x:(x+dx), :])

                
