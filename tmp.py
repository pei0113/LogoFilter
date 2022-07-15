import os
import cv2
import shutil
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters.rank import equalize
from skimage.morphology import disk


# q_root = r'C:\Users\pei.liu\Downloads\Data_510\DeleteBkg\low'
# s_root = r'C:\Users\pei.liu\Downloads\Data_510\Raw'
#
#
# for _root, dirs, files in os.walk(q_root):
#     for file in files:
#         if file.endswith('png'):
#             fn_ori = file.split('_dbkg')[0]+'.png'
#             path_src = os.path.join(s_root, fn_ori)
#             path_dst = os.path.join(q_root, file)
#             shutil.move(path_src, path_dst)

db_root = r'C:\Users\pei.liu\Downloads\Data_510\DeleteBkg\high'
selem = disk(20)
img_eqs = []
for _root, dirs, files in os.walk(db_root):
    for file in files:
        if file.endswith('png'):
            img = cv2.imread(os.path.join(_root, file), 0)
            img_eq = equalize(img.astype('uint8'), selem=selem)
            img_eqs.append(img_eq)
            # plt.subplot(121)
            # plt.imshow(img, 'gray')
            # plt.subplot(122)
            # plt.imshow(img_eq, 'gray')
            # plt.show()

img_eqs = np.array(img_eqs)
img_eq_mean = np.mean(img_eqs, axis=0)
mean_eq = equalize(img_eq_mean.astype('uint8'), selem=selem)
plt.subplot(121)
plt.imshow(img_eq_mean, 'gray')
plt.subplot(122)
plt.imshow(mean_eq, 'gray')
plt.show()
