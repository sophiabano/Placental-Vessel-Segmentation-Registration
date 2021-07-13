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

from segmentation_models import Unet
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
import argparse
import shutil, os, math, random
from time import time
from keras import optimizers
import numpy as np
from skimage.transform import resize
import scipy.io as sio
from sklearn.metrics import confusion_matrix

from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint,TensorBoard

from utilsSB import evaluate_segmentation
from data import *
from model_unet import unet_vanilla


#dynamically grow memory help from: https://kobkrit.com/using-allow-growth-memory-option-in-tensorflow-and-keras-dc8c8081bc96
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=800, help='Number of epochs to train for')
parser.add_argument('--mode', type=str, default="train", help='Select "train", or "predict" or test') 
parser.add_argument('--foldno', type=str, default="anon001", help='5 folds. anon001 means to not use anon001 data for training"') 
parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')
parser.add_argument('--batch_size', type=int, default=16, help='Number of images in each batch')
parser.add_argument('--losstype', type=str, default="BCE_jaccard", help='BCE or BCE_jaccard ')
parser.add_argument('--backbone', type=str, default="resnet101", help='vgg16 or resnet34 or resnet50 or resnet101 or unet_vanilla')

xx = random.randint(1,1000)
args = parser.parse_args()

data_path = "dataset/"

tag = args.foldno
EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size

train_str = ['anon001','anon002','anon003','anon005', 'anon010','anon012']

if args.mode == 'train':  
    if tag in train_str: train_str.remove(tag)
    # Creating training data folder for a particular fold  
    if os.path.isdir(data_path+tag+str(xx)+'_traintemp'):
        shutil.rmtree(data_path+tag+str(xx)+'_traintemp')
    for i in range(len(train_str)):
        copytree(data_path+train_str[i]+'/images',data_path+tag+str(xx)+'_traintemp/images')
    for i in range(len(train_str)):
        copytree(data_path+train_str[i]+'/masks',data_path+tag+str(xx)+'_traintemp/masks')      
    IMG_COUNT = len(os.listdir(data_path+tag+str(xx)+'_traintemp/images'))  
    IMG_COUNT_VAL = len(os.listdir(data_path+tag+'/images'))


BACKBONE = args.backbone
preprocess_input = None

data_gen_args = dict(rotation_range=30,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.1,
                    zoom_range=0.1,
                    horizontal_flip=True,
                    vertical_flip=True,
                    brightness_range = (0.8, 1.2),
                    fill_mode='nearest'
                    )

TrainData = trainGenerator(BATCH_SIZE,data_path+tag+str(xx)+"_traintemp/",'images','masks',preprocess_input, data_gen_args,save_to_dir = None)

data_gen_args_v = dict()
ValData = valGenerator(BATCH_SIZE,data_path+tag+"/",'images','masks',preprocess_input, data_gen_args_v,save_to_dir = None)

# define model
if BACKBONE == 'vgg16':
    model = Unet(BACKBONE,classes=1, encoder_weights='imagenet') #imagenet'
elif BACKBONE == 'resnet34':
    model = Unet(BACKBONE,classes=1, encoder_weights='imagenet') #imagenet'
elif BACKBONE == 'resnet50':
    model = Unet(BACKBONE,classes=1, encoder_weights='imagenet') #imagenet'
elif BACKBONE == 'resnet101':
    model = Unet(BACKBONE,classes=1, encoder_weights='imagenet') #imagenet'      
elif BACKBONE == 'unet_vanilla':
    model = unet_vanilla()
  
    
model.summary()
if args.continue_training:
    print('continue training')
    model.load_weights('checkpoints/best_unet_resnet101_anon010_Isz448_cp224_Ino445_BCE_jaccard_lr_0.0001_256_912_0.76_0.54.hdf5')
    
lr = 0.0003
adam = optimizers.Adam(lr) 
plot_model(model, to_file=BACKBONE+'.png', show_shapes=True)

if args.losstype == 'BCE_jaccard':
    model.compile(optimizer=adam, loss=bce_jaccard_loss, metrics=[iou_score])
    print("BCE_jaccard")
elif args.losstype == 'BCE':
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[iou_score])
    print("BCE")   
elif args.losstype == 'WCE_sadda':
    sgd = optimizers.SGD(lr, momentum=0.9)
    model.compile(optimizer=sgd, loss=[WCE_Sadda_loss(weight_val=3000)], metrics=[iou_score])
    print("WCE_Sadda")
        

if args.mode == "train":
    print("\n***** Begin training *****")
    print("Backbone -->", BACKBONE)
    print("Model --> Unet")
    print("FOLD -->", tag)
    print("Loss type -->", args.losstype)    
    
    tb_dir = "logs/{}".format(time())+'unet_'+BACKBONE+'_'+tag+'_TIMG_'+str(IMG_COUNT)+'_'+str(xx)
    tensorboard = TensorBoard(log_dir=tb_dir, write_graph=True)#, update_freq=100)
    #Best checkpoint
    model_checkpoint = ModelCheckpoint('./checkpoints/best_unet_'+BACKBONE+'_'+tag+'_ext_simdataP4_Isz448_cp224_Ino'+str(IMG_COUNT)+'_'+args.losstype+'_lr_'+str(lr)+'_'+str(xx)+"_{epoch:02d}_{iou_score:.2f}_{val_iou_score:.2f}"+'.hdf5', monitor='loss',verbose=1, save_best_only=True)
    #latest checkpoint
    model_checkpoint2 = ModelCheckpoint('./checkpoints/latest_unet_'+BACKBONE+'_'+tag+'_ext_simdataP4_Isz448_cp224_Ino'+str(IMG_COUNT)+'_'+args.losstype+'_lr_'+str(lr)+'_'+str(xx)+'.hdf5', monitor='loss',verbose=1, save_best_only=False)
       
    callbacks_list = [model_checkpoint, model_checkpoint2, tensorboard]#, image_history]#, lrate]

    model.fit_generator(TrainData,steps_per_epoch=IMG_COUNT/BATCH_SIZE,epochs=EPOCHS,validation_data=ValData, validation_steps=IMG_COUNT_VAL/BATCH_SIZE, shuffle=True, callbacks=callbacks_list)
    
    shutil.rmtree(data_path+tag+str(xx)+'_traintemp')

###############################################################################    
if args.mode == "predict" or args.mode == "test":
    save_output_fig = False
    chk_name = "anon012f/best_unet_resnet101_anon012_Isz448_cp224_Ino385_BCE_jaccard_lr_0.0003_458_192_0.68_0.57"
    model.load_weights('checkpoints/' + chk_name + '.hdf5')
    
    if not os.path.isdir("%s/%s/%s/%s"%("results",tag,chk_name,"vis")):
        os.makedirs("%s/%s/%s/%s"%("results", tag, chk_name,"vis"))

    if not os.path.isdir("%s/%s/%s/%s"%("results",tag,chk_name,"predicted_mask")):
        os.makedirs("%s/%s/%s/%s"%("results", tag, chk_name,"predicted_mask"))
            
    test_img_files = []
    test_mask_files = []
    filenames = [];
    for file in np.sort(os.listdir(data_path + tag + '/images/')):
        cwd = os.getcwd()
        test_img_files.append(data_path + tag + '/images/'+file)
        filenames.append(file)

        
    if args.mode == "predict":
        for file in np.sort(os.listdir(data_path+ tag + '/masks/')):
            cwd = os.getcwd()
            test_mask_files.append(data_path + tag + '/masks/'+file)
         
    accuray_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    iou_list = [] 
    mean_iou_list = []
    target_size = [448,448]
    for file_cnt in range(len(test_img_files)):
        img = io.imread(test_img_files[file_cnt])
        img = resize(img, (target_size[0], target_size[1], img.shape[2] ),
                       anti_aliasing=True)
        img = img #/ 255
        img = np.reshape(img,(1,)+img.shape)
        #img = preprocess_input(img)
        
        if args.mode == "predict":
            mask = io.imread(test_mask_files[file_cnt])
            mask = resize(mask, (target_size[0], target_size[1], mask.shape[2] ),
                   anti_aliasing=True)
            mask = np.mean(mask,axis=2)
            mask = (mask >0).astype(np.int)
        
        img_predict_out = model.predict(img)
        img_predict_out = np.squeeze(img_predict_out)
        
        img_predict = np.copy(img_predict_out)
        img_predict[img_predict_out > 0.5] = 1
        img_predict[img_predict_out <= 0.5] = 0

        if args.mode == "predict":
            accuracy, prec, rec, f1, mean_iou, iou = evaluate_segmentation(pred=img_predict, label=mask, num_classes=1)
            
            accuray_list.append(accuracy)
            precision_list.append(prec)
            recall_list.append(rec)
            f1_list.append(f1)
            mean_iou_list.append(mean_iou)
            iou_list.append(iou)
            
            flat_pred = img_predict.flatten()
            flat_label = mask.flatten()
            
            confm = confusion_matrix(flat_pred, flat_label)
            
        if save_output_fig: 
            plt.subplot(1, 3, 1)
            plt.imshow(np.squeeze(img))
            plt.title('Input')
            if args.mode == "predict":
                plt.subplot(1, 3, 2)
                plt.imshow(mask,vmin=0, vmax=1)
                plt.title('gt')
                plt.subplot(1, 3, 3)
                plt.imshow(img_predict,vmin=0, vmax=1)
                plt.title('Pred iou: %.2f' %iou)
            if args.mode == "test":
                plt.subplot(1, 3, 2)
                plt.imshow(np.squeeze(img))#,'gray', interpolation='none')
                plt.imshow(img_predict, 'jet', interpolation='none', alpha=0.4)   
                plt.subplot(1, 3, 3)
                plt.imshow(img_predict,vmin=0, vmax=1)
                plt.title('Pred')
            
            plt.savefig("%s/%s/%s/%s/%s"%("results",tag,chk_name,"vis", filenames[file_cnt]))
            io.imsave("%s/%s/%s/%s/%s"%("results", tag,chk_name,"predicted_mask", filenames[file_cnt]),img_predict_out)
            plt.show()
            
    if args.mode == "predict": 
        avg_acc = np.mean(accuray_list)
        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_f1 = np.mean(f1_list)
        avg_mean_iou = np.mean(mean_iou_list)
        avg_iou = np.mean(iou_list)
        
        std_acc = np.std(accuray_list)
        std_precision = np.std(precision_list)
        std_recall = np.std(recall_list)
        std_f1 = np.std(f1_list)
        std_mean_iou = np.std(mean_iou_list)
        std_iou = np.std(iou_list)
        
        print("Average accuracy = ", avg_acc)
        print("Std accuracy = ", std_acc)
        
        print("Average precision = ", avg_precision)
        print("Std precision = ", std_precision)
        
        print("Average recall = ", avg_recall)
        print("Std recall = ", std_recall)
        
        print("Average F1 = ", avg_f1)
        print("Std F1 = ", std_f1)

        print("Average mean IoU = ", avg_mean_iou)
        print("Std mean IoU = ", std_mean_iou)
        
        print("Average IoU = ", avg_iou)
        print("Std IoU = ", std_iou)
        
        sio.savemat('./checkpoints/'+chk_name+tag+'.mat', {'accuracy': accuray_list,
                              'precision':precision_list, 
                              'recall': recall_list,
                              'f1score': f1_list,
                              'mean_iou': mean_iou_list,
                              'iou': iou_list})
    

