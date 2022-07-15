import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


data_root = r'C:\Users\pei.liu\Downloads\Data_510\Raw'

fns = []
imgs = []
for _root, dirs, files in os.walk(data_root):
    for file in files:
        if file.endswith('png'):
            fn = os.path.join(_root, file)
            fns.append(fn)
            img = cv2.imread(fn, 0)
            imgs.append(img)

imgs = np.array(imgs)
img_logo = np.mean(imgs, axis=0)

for i in range(0, len(imgs)):
    img = imgs[i]
    fn = fns[i]
    # calculate RV
    img_mean = np.mean(img)
    img_std = np.std(img)
    img_R = np.where(img >= img_mean, 1, 0)
    img_V = np.where(img < img_mean, 1, 0)
    value_R = np.mean(img[img_R])
    value_V = np.mean(img[img_V])
    value_RV = abs(value_R - value_V)
    print("R:{:.2f}, V:{:.2f}, RV:{:.2f}".format(value_R, value_V, value_R-value_V))
    # plt.subplot(121)
    # plt.imshow(img_R, 'gray')
    # plt.subplot(122)
    # plt.imshow(img_V, 'gray')
    # plt.show()
    img_sub = img - img_logo
    img_sub = img_sub + (np.mean(img)-np.mean(img_sub))
    img_sub[img_sub > 255] = 255
    img_sub[img_sub < 0] = 0
    # if value_RV < 0.1:
    #     cv2.imwrite(fn.replace('Raw\\', 'DeleteBkg\\low\\').replace('.png', '_dbkg_rv{:.2f}.png'.format(value_RV)), img_sub.astype('uint8'))
    # else:
    #     cv2.imwrite(fn.replace('Raw\\', 'DeleteBkg\\high\\').replace('.png', '_dbkg_rv{:.2f}.png'.format(value_RV)), img_sub.astype('uint8'))

    print("img_mean={:.2f}, logo_mean={:.2f}, sub_mean={:.2f}".format(np.mean(img), np.mean(img_logo), np.mean(img_sub)))
    plt.subplot(221)
    plt.imshow(img, 'gray', vmin=0, vmax=255)
    plt.subplot(222)
    plt.imshow(img_logo, 'gray', vmin=0, vmax=255)
    plt.subplot(223)
    plt.imshow(img_sub, 'gray', vmin=0, vmax=255)
    plt.show()
