import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from skimage.filters.rank import equalize
from skimage.morphology import disk

data_root = r'C:\Users\pei.liu\Downloads\Data_510\Raw'

fns = []
img_path = []
img_raw = []
for _root, dirs, files in os.walk(data_root):
    for file in files:
        if file.endswith('png'):
            fn = os.path.join(_root, file)
            fns.append(fn)
            img_path.append(fn)
            img = cv2.imread(fn, 0)
            img_raw.append(img)

rv_list, bds = [], []
bds_cnt = 0

for ii in range(len(img_path)):
    img = np.int16(img_raw[ii])

    img_highpass = img - cv2.blur(img, (3, 3))
    var = np.mean(np.abs(img_highpass))
    rv_list.append(var)

##plt.plot(rv_list, '.')
##plt.show()

    if 2.65 < var < 3.5:
##        img_bandpass = cv2.blur(img, (3, 3)) - cv2.blur(img, (7, 7))
        bds.append(img)
        bds_cnt += 1

img_logo = np.mean(bds, axis=0)
selem = disk(20)
eq_logo = equalize(img_logo.astype('uint8'), selem=selem)
cv2.imwrite('logo_.png', ((img_logo-img_logo.min())*255/(img_logo.max()-img_logo.min())).astype('uint8'))
plt.subplot(121)
plt.imshow(img_logo, 'gray', vmin=0, vmax=255)
plt.subplot(122)
plt.imshow(eq_logo, 'gray', vmin=0, vmax=255)
plt.show()

# for ii, img in enumerate(img_raw):
#     rv = rv_list[ii]
#     fn = img_path[ii]
#     img_sub = img - img_logo
#     img_sub = img_sub + (np.mean(img) - np.mean(img_sub))
#     img_sub[img_sub > 255] = 255
#     img_sub[img_sub < 0] = 0
    # print(rv)
    # plt.subplot(221)
    # plt.imshow(img, 'gray')
    # plt.subplot(222)
    # plt.imshow(img_logo, 'gray')
    # plt.subplot(223)
    # plt.imshow(img_sub, 'gray')
    # plt.show()

    # if rv < 3:
    #     cv2.imwrite(fn.replace('Raw\\', 'DeleteBkg\\low\\').replace('.png', '_dbkg_rv{:.2f}.png'.format(rv)), img)
    # else:
    #     cv2.imwrite(fn.replace('Raw\\', 'DeleteBkg\\high\\').replace('.png', '_dbkg_rv{:.2f}.png'.format(rv)), img_sub.astype('uint8'))
