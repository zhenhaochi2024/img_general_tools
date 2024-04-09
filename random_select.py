import random

src = r'/media/police1/buhaochi/cv/SIDD_medium/SIDD_Medium_Srgb/dataset_InvD'
desc = r'/media/police1/buhaochi/dataset_all/CV/denoise_SIDD'

gt_src = src + '/GT/'
noise_src = src + "/Noisy/"

gt_desc = desc + '/gt/'
noise_desc = desc + "/noised/"

import os
import shutil
for i in range(20):
    a = random.randint(0,320)
    b = random.randint(0,99)
    file = str(a) + '_' + str(b) + '.PNG'
    gt_path = gt_src + file
    n_path = noise_src +file
    gt_out = gt_desc + file
    n_out = noise_desc + file
    shutil.copy(gt_path,gt_out)
    shutil.copy(n_path,n_out)

