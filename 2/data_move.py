import os
import cv2.cv2 as cv2
import shutil
src = '/Users/liang-yulong/Desktop/dataset_kaggledogvscat/train/'
dst1 = '/Users/liang-yulong/Desktop/dataset_kaggledogvscat/dog/'
dst2 = '/Users/liang-yulong/Desktop/dataset_kaggledogvscat/cat/'
f = os.listdir(src)
for fl in f:
    if fl.split('.')[0] == 'dog':
        src_path = os.path.join(src,fl)
        dst1_path = os.path.join(dst1,fl)
        shutil.move(src_path,dst1_path)
    else:
        src_path = os.path.join(src,fl)
        dst2_path = os.path.join(dst2,fl)
        shutil.move(src_path,dst2_path)
    