"""
@author: Sophia Bano and Oluwatosin Alabi

This is the implmentation of the paper 'Deep placental Vessel Segmentation for 
Fetoscopic Mosaicing' which was presented at MICCAI2020.

Note: If you use this code, consider citing the following paper: 
    
Bano, S., Vasconcelos, F., Shepherd, L.M., Vander Poorten, E., Vercauteren, T., 
Ourselin, S., David, A.L., Deprest, J. and Stoyanov, D., 2020, October. 
Deep placental vessel segmentation for fetoscopic mosaicking. In International 
Conference on Medical Image Computing and Computer-Assisted Intervention 
(pp. 763-773). Springer, Cham.
"""


import matplotlib.pyplot as plt
import cv2, os
import numpy as np
from PIL import Image
from pathlib import Path


# configuration dictionary to work with video sequences from MICCAI2020 placental data.
seq_exact = {
    "anon001": {
        "name": "anon001",
        "file_length": 100,   # for miccai anon001 filelength 400, start 20
        "start": 0,
        "v_crop_top": 0,
        "v_crop_bottom": 0,
    },
    "anon002": {
        "name": "anon002",
        "file_length": 200,
        "start": 120,
        "v_crop_top": 0,
        "v_crop_bottom": 0,
    },
    "anon003": {
        "name": "anon003",
        "file_length": 50,
        "start": 10,
        "v_crop_top": 0,
        "v_crop_bottom": 0,
    },
    "anon005": {
        "name": "anon005",
        "file_length": 100,
        "start": 20,
        "v_crop_top": 60,
        "v_crop_bottom": 20,
    },
    "anon010": {
        "name": "anon010",
        "file_length": 100,
        "start": 70,
        "v_crop_top": 80,
        "v_crop_bottom": 70,
    },
    "anon012": {
        "name": "anon006",
        "file_length": 100,
        "start": 351,
        "v_crop_top": 80,
        "v_crop_bottom": 70,
    }
}

def readHfromTXT(miccai_Hpath):
    """
    :param miccai_Hpath: Text files containing H (homography) matrix for each consecutive image pairs in the video sequence
    :return: returns H_array - a numpy array of size (N,3,3) where N is the number of images/homography matrices
    """
    fullTxtPaths =  [ miccai_Hpath + '/' + f  for f  in sorted(os.listdir(miccai_Hpath))]
    
    Fileopen = open(fullTxtPaths[0], 'r').read().strip()
    Hstr3 = Fileopen.split("\n")
    
    H_indv = np.array([])
    for H1 in Hstr3:
        Hstr1 = H1.split(" ")
        Hfloat = [float(s) for s in Hstr1]
        H_indv = np.append(H_indv, Hfloat)
    #print(H_indv)
        
    H_array = np.array([H_indv.reshape(3,3)])
        
    for i in range(len(fullTxtPaths)-1):
        Fileopen = open(fullTxtPaths[i+1], 'r').read().strip()
        Hstr3 = Fileopen.split("\n")
        
        H_indv = np.array([])
        for H1 in Hstr3:
            Hstr1 = H1.split(" ")
            Hfloat = [float(s) for s in Hstr1]
            H_indv = np.append(H_indv, Hfloat)
        #print(H_indv)
            
        H_indv =np.array([H_indv.reshape(3,3)])
    
        H_array  = np.append(H_array,H_indv, axis = 0)
    return H_array
    
def get_mask_im(fullImgPaths, mask_path, crop_top, crop_bottom):
    """
    :param fullImgPaths: Image path of images to be processed, need only the size of one image there
    :param mask_path: path to the mask image to be used
    :param crop_top: the amount of pixels to be removed at the top due to dead pixels
    :param crop_bottom:  the amount of pixels to be removed at the bottom due to dead pixels
    :return: returns mask image
    """
    img_1 = cv2.imread(fullImgPaths[0])
    img_1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2RGB)


    mask_im = Image.open(mask_path)

    mask_im = mask_im.resize((img_1.shape[0], img_1.shape[1]), Image.ANTIALIAS)
    mask_im = np.array(mask_im)
    mask_im = mask_im * np.uint8(255)

    # crop mask
    mask_im[:crop_top] = 0
    mask_im[(mask_im.shape[0] - crop_bottom):] = 0

    mask_im = cv2.resize(mask_im, img_1.shape[1::-1])

    return mask_im

def getHGlobal(H_array, fullImgPaths, middle_num):
  """
  Get global H wrt a frame called middle_num
  :param H_array: pair to pair transformations
  :param fullImgPaths: Image path
  :param middle_num: frame to be used as origin
  :return: returns global registration wrt to frame given by middle num.
  """
  H_global = np.zeros((len(H_array), 3,3 ))

  for i in range(len(H_array)):
    # print(i)
    H_intermediate = np.eye(3)
    if i < middle_num:
      for j in range(i, int(middle_num)):
         H_intermediate = np.matmul( H_intermediate,
                                    np.vstack([H_array[j], [0,0,1]]) )
      H_intermediate = np.linalg.inv(H_intermediate)
    elif i > middle_num:
      for j in range(int(middle_num), i):
         H_intermediate = np.matmul( H_intermediate,
                                    np.vstack([H_array[j], [0,0,1]]) )

    #note that if i == middle_num, we use unity array
    H_global[i] = H_intermediate

  return H_global

def globalRegistration(img, imageName, index, padding, mask_im, H_global):
    ht, wd, cc= img.shape

    ww = wd + (2*padding)
    hh = ht + (2*padding)

    xx = (ww - wd) // 2
    yy = (hh - ht) // 2

    color = (0,0,0)

    img = cv2.bitwise_and(img, img, mask=mask_im)
    
    # vis.visualizeImg(img)

    T = np.copy(H_global[index])

    T[:2, 2] = T[:2, 2] + [xx , yy] 
    # print(T)

    result = cv2.warpPerspective(img, T, (ww, hh))

    # vis.visualizeImg(result)

    return result

def getTransparentImg(src, imageName, seq_name, writepath):
  """
  Get transparent images from non transparent ones
  :param src: image to be made transparent
  :param imageName: name it should be stored as
  :return: returns transparent image
  """
  tmp = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
  _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
  r, g, b = cv2.split(src)
  rgba = [b,g,r, alpha]
  dst = cv2.merge(rgba,4)
  save_name = writepath + '/' + seq_name + '/' + imageName
  cv2.imwrite(save_name, dst)

def do_global_registration(fullImgPaths, middle_num, seq_name, seq_length, seq_start, padding_size, mask_im, H_global, writepath):
    for i in range((seq_length)):
        imgPath = fullImgPaths[i + seq_start - 1];
        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 
        imageName = Path(fullImgPaths[i + seq_start - 1]).name

        src = globalRegistration(img, imageName, i, padding_size, mask_im, H_global)
        getTransparentImg(src, imageName, seq_name , writepath)
        
def visualizeImg(img):
  """
  Visualize an image, when I have the image in matrix form
  :param img:
  :return:
  """
  plt.figure(figsize=(10,4))
  plt.imshow(img)
  plt.show()
  
def visualizeStitch(srcImg, destImg, H, padding, transformation, mask_im, showImages=True):
    """
    :param srcImg: srcImg
    :param destImg: destImg
    :param H: transformation matrix
    :param padding: padding value if used
    :param transformation: homography or affine
    :param mask_im: mask image
    :param showImages: display flag
    :return: none
    """
    ht, wd, cc = destImg.shape

    ww = wd + (2 * padding)
    hh = ht + (2 * padding)

    xx = (ww - wd) // 2
    yy = (hh - ht) // 2

    if transformation == "Homography":
        result = cv2.warpPerspective(srcImg, H, (ww, hh))
    elif transformation == "Affine":
        result = cv2.warpAffine(srcImg, H, (ww, hh))

    alpha_s = mask_im / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        result[yy:yy + ht, xx:xx + wd, c] = (alpha_s * destImg[:, :, c] +
                                             alpha_l * result[yy:yy + ht, xx:xx + wd, c])

    print("Visualize stitch")
    if showImages:
        plt.figure(figsize=(10, 4))
        plt.imshow(result)

        plt.show()
        
def generate_video(image_folder, video_name, video_frames_path):
    """
    Generate videos from warped images with lots of paddings.
    :param image_folder: folder path
    :param video_name: name for video
    :param video_frames_path: path to saved video.
    :return: none
    """
    
    try:
        os.stat(video_frames_path)
    except:
        os.makedirs(video_frames_path)
    
    images = [img for img in os.listdir(image_folder)
              if img.endswith(".jpg") or
              img.endswith(".jpeg") or
              img.endswith("png") or
              img.endswith("tif")]

    images.sort()

    print(images)

    frame = cv2.imread(os.path.join(image_folder, images[0]))

    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_frames_path + '/' + video_name, fourcc, 1, (width, height))

    # Appending the images to the video one by one
    video_frame = np.zeros((height, width, 3), np.uint8)
    for image in images:
        img = cv2.imread(os.path.join(image_folder, image), cv2.IMREAD_UNCHANGED)
        video_frame = overlay_transparent(video_frame, img)
        cv2.imwrite(os.path.join(video_frames_path, image), video_frame)
        video.write(video_frame)

        # Deallocating memories taken for window creation
    cv2.destroyAllWindows()
    video.release()  # releasing the video generated

def overlay_transparent(bg_img, img_to_overlay_t):
    """
    Overlay new image on background only in positions where warped image is.
    :param bg_img: background image
    :param img_to_overlay_t: new image with transparent background
    :return:
    """
    # Extract the alpha mask of the RGBA image, convert to RGB
    b,g,r,a = cv2.split(img_to_overlay_t)
    overlay_color = cv2.merge((b,g,r))

    #reduce size of image
    # bg_img = get_square_in_image(bg_img)

    # Black-out the area behind the logo in our original ROI
    # img1_bg = cv2.bitwise_and(bg_img.copy(), bg_img.copy(), mask = cv2.bitwise_not(a))
    img1_bg = cv2.bitwise_and(bg_img, bg_img, mask=cv2.bitwise_not(a))

    # Mask out the logo from the logo image.
    img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask = a)

    # Update the original image with our new ROI
    bg_img = cv2.add(img1_bg, img2_fg)

    return bg_img