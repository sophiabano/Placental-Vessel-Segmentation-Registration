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
import cv2
import numpy as np
import os
import argparse


from utilsReg import *

parser = argparse.ArgumentParser(description='create video files from frames')
parser.add_argument("--video_seq_name", type=str, default = "anon001", help="video sequence name")
parser.add_argument("--videoframes_path", type=str, default = "sample_data", help="path to folder containing video frames (in folder 'images') and fetoscope masks (in folder 'mask')")
parser.add_argument("--miccai2020fetreg_txt", type=str, default = 'sample_data/Vessel-based_Hpred_plus_100/anon001', help="path of folder containing individual text files with Homography martix (output of the registration algorithm)")

writepath = 'MosaicVisVideo' # output mosaic visualisation written to this path


args = parser.parse_args()

seq_name = args.video_seq_name
frames_path = args.videoframes_path + '/' + seq_name
fullImgDirPath = frames_path + '/images'
mask_path = frames_path + '/mask/' + seq_name + '_mask.png'
miccai_Hpath = args.miccai2020fetreg_txt


transformation = "Affine"
padding_size = 2000
showImages = False

window_size = 1
frame_distance = 5
 

fullImgPaths =  [ fullImgDirPath + '/' + f  for f  in sorted(os.listdir(fullImgDirPath))]

v_crop_top = seq_exact[seq_name]["v_crop_top"]
v_crop_bottom = seq_exact[seq_name]["v_crop_bottom"]

mask_im = get_mask_im(fullImgPaths, mask_path, v_crop_top, v_crop_bottom)


seq_length = seq_exact[seq_name]["file_length"] 
seq_start = seq_exact[seq_name]["start"] 

# Read H from MICCAI2020 registration algorithm output
H_array = readHfromTXT(miccai_Hpath)
H_array = H_array[seq_start:seq_start+seq_length,:,:]

## Get registration
try:
    os.stat(writepath + '/' + seq_name)
except:
    os.makedirs(writepath + '/' + seq_name)

# Get affine matrix
H_affine = np.zeros( (len(H_array), 2,3))
for i in range(len(H_array)):
  H = H_array[i,:,:]
  H_affine[i] =  H[:2, :]
  
  
# Aligning to the middle frame
middle_num = seq_length/2 #len(fullImgPaths)//2
middle_num = middle_num - (middle_num%3)
H_global = getHGlobal(H_affine, fullImgPaths, middle_num)

# Plotting glocal registration onto images
do_global_registration(fullImgPaths, middle_num, seq_name, seq_length, seq_start, padding_size, mask_im, H_global, writepath)

# Calling the generate_video function 
generate_video(writepath + '/' + seq_name, seq_name + ".MP4", writepath + '/' + seq_name + "/mosaic")




