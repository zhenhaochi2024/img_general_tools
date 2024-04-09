import os
import cv2 as cv
import PIL.Image as Image
import numpy as np

'''
region:(x,y,d_x,d_y):x for height; y for width
scale:(height,width)
'''
def enlarge(img_in,img_out,region,scale=None):

    img = cv.imread(img_in)

    H = img.shape[0]
    W = img.shape[1]

    if scale == None:
        scale = (min(H,W)//3,min(H,W)//3)
    height, width = scale

    x, y, d_x, d_y = region
    part = img[x:x+d_x, y:y+d_y]

    mask = cv.resize(part, (width, height), fx=0, fy=0, interpolation=cv.INTER_LINEAR)
    img[0:height, W-width:W] = mask

    # 画框并连线
    cv.rectangle(img, (y, x), (y+d_y, x+d_x), (0, 255, 255), 2)
    cv.rectangle(img, (W-width, 0), (W - 2, height), (0, 255, 255), 2)

    cv.imwrite(img_out,img)

# img = r'1.jpg'
# img_out = r'a.jpg'
# # region = (350,300,50,50)
# region = (100,200,50,50)
# scale = (200,100)
# enlarge(img,img_out,region)

src_dir = r'/media/police1/buhaochi/dataset_all/CV/denoise_SIDD/experiment'
desc_dir = r'/media/police1/buhaochi/dataset_all/CV/denoise_SIDD/enlarge'

for dir in os.listdir(src_dir):
    class_dir = os.path.join(src_dir,dir)
    desc_class_dir = os.path.join(desc_dir,dir)
    if not os.path.exists(desc_class_dir):
        os.mkdir(desc_class_dir)
    for dir in os.listdir(class_dir):
        img_dir = os.path.join(class_dir,dir)
        desc_img_dir = os.path.join(desc_class_dir, dir)
        if not os.path.exists(desc_img_dir):
            os.mkdir(desc_img_dir)
        for img in os.listdir(img_dir):
            img_path = os.path.join(img_dir,img)
            desc_path = os.path.join(desc_img_dir,img)
            data = np.array(Image.open(img_path))
            H,W,C = data.shape
            region = (W//2, H//2, W//8, H//8)
            scale = (min(H,W)*6//12,min(H,W)*6//12)
            enlarge(img_path, desc_path, region)
