import PIL.Image as Image
import os

dir = r'/media/police1/buhaochi/dataset_all/CV/denosing_other/dataset/UCMerced_LandUse_sparseresidential'
tifs = os.listdir(dir)
out = r'/media/police1/buhaochi/dataset_all/CV/denosing_other/dataset/sparseresidential'
print(tifs)
for tif in tifs:
    path = os.path.join(dir,tif)
    res_path = os.path.join(out,tif.split('.')[0]+'.jpg')
    img = Image.open(path)
    img.save(res_path)
