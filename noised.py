import os
import glob
import h5py
from PIL import Image
import numpy as np
import cv2 as cv

def get_img_list(img_dir):
    types = ('*.bmp', '*.png','*.tif')
    files = []
    for tp in types:
        files.extend(glob.glob(os.path.join(img_dir, tp)))
    files.sort()
    return files


data_dir = r'/media/police1/buhaochi/dataset_all/CV/denosing_other/resize'
dirs = os.listdir(data_dir)
sigma = 15/255.0

noised_dir = r'/media/police1/buhaochi/dataset_all/CV/denosing_other/noised_15'
for dir in dirs:
    in_dir = os.path.join(data_dir,dir)
    out_dir = os.path.join(noised_dir,dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    img_list = os.listdir(in_dir)
    for img in img_list:
        if img[-4:] == "jfif" or img[-3:] == "tif":
            continue
        img_path = os.path.join(in_dir,img)
        img_data = cv.imread(img_path)  # if use CV2.imread(img),will get a "BGR" img array
        img_data = np.array(img_data).astype(np.float32)  # need to transform to ndarray
        # if img_data.shape[2] != 3:
        #     continue
        # img_data = img_data[:512, :512, :]
        img_data = np.float32(img_data / 255.0)
        noise = np.random.normal(0, sigma, img_data.shape)
        noised = img_data + noise
        noised = noised * 255
        out_path = os.path.join(out_dir,img)
        cv.imwrite(out_path,noised)

