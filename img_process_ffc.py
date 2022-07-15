import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters.rank import equalize
from skimage.morphology import disk


selem = disk(20)
img_eqs = []
db_root = r'C:\Users\pei.liu\Downloads\Data_510_FFC_BeforeAfter\Raw'
for _root, dirs, files in os.walk(db_root):
    for file in files:
        if file.endswith('.png'):
            path_raw = os.path.join(_root, file)
            path_ffc = path_raw.replace('\\Raw\\', '\\DeleteBkg\\high\\').replace('.png', '_ffc.png')
            img_raw = cv2.imread(path_raw, 0)
            img_ffc = cv2.imread(path_ffc, 0)

            # plot raw / ffc
            # plt.subplot(121)
            # plt.imshow(img_raw, 'gray')
            # plt.subplot(122)
            # plt.imshow(img_ffc, 'gray')
            # plt.show()

            img_eq = equalize(img_ffc.astype('uint8'), selem=selem)
            img_eqs.append(img_eq)

img_eqs = np.array(img_eqs)
img_eq_mean = np.mean(img_eqs, axis=0)
mean_eq = equalize(img_eq_mean.astype('uint8'), selem=selem)
plt.subplot(121)
plt.imshow(img_eq_mean, 'gray')
plt.subplot(122)
plt.imshow(mean_eq, 'gray')
plt.show()