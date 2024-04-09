import PIL.Image as Image
import os

import numpy as np

import random

# img_dir = r'/media/police1/buhaochi/cv/denoise_dataset/expriment/real_noise/RNI15'
#
# for img in os.listdir(img_dir):
#     data = np.array(Image.open(os.path.join(img_dir,img)))
#     if data.shape[0] < 320 or data.shape[1] < 320:
#         data = data[:256, :256]
#     else:
#         data = data[:320, :320]
#     # data = data[:320,:320]
#     out_img = Image.fromarray(np.uint8(data))
#     out_img.save(os.path.join(img_dir,img))

src_dir = r'/media/police1/buhaochi/dataset_all/CV/denoise_SIDD/dataset'
desc_dir = r'/media/police1/buhaochi/dataset_all/CV/denoise_SIDD/resize'


for dir in os.listdir(src_dir):
    img_dir = os.path.join(src_dir,dir)
    desc_img_dir = os.path.join(desc_dir, dir)
    if not os.path.exists(desc_img_dir):
        os.mkdir(desc_img_dir)
    for img in os.listdir(img_dir):
        if img[-4:] == "jfif" or img[-3:] == "tif":
            continue
        img_path = os.path.join(img_dir,img)
        desc_path = os.path.join(desc_img_dir,img)
        data = np.array(Image.open(img_path))
        if len(data.shape) == 2:
            continue
        H,W,C = data.shape
        if H>320 and W > 320:
            # h = random.randint(0,H-320-1)
            # w = random.randint(0,W-320-1)
            h=0
            w=0
            data = data[h:h+320,w:w+320]
        elif H>=256 and W >=256:
            h = random.randint(0,H-256)
            w = random.randint(0,W-256)
            data = data[h:h+256,w:w+256]
        elif H>128 and W > 128:
            h = random.randint(0, H - 128 - 1)
            w = random.randint(0, W - 128 - 1)
            data = data[h:h + 128, w:w + 128]
        else:
            h = random.randint(0, H - 64 - 1)
            w = random.randint(0, W - 64 - 1)
            data = data[h:h + 64, w:w + 64]
        out_img = Image.fromarray(np.uint8(data))
        out_img.save(desc_path)


