import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# data_root = r'C:\Users\pei.liu\Downloads\Data_510\Raw'
# fns = []
# imgs = []
# for _root, dirs, files in os.walk(data_root):
#     for file in files:
#         if file.endswith('png'):
#             fn = os.path.join(_root, file)
#             fns.append(fn)
#             img = cv2.imread(fn, 0)
#             imgs.append(img)
# img = imgs[10]
img = cv2.imread(r'C:\Users\pei.liu\Downloads\Data_510\Raw\RawFrame_20220705_184502.554.png', 0)
# img = cv2.imread(r'C:\Users\pei.liu\Downloads\Lenna.jpg', 0)

# 傅立葉轉換
# 二維快速傅立葉轉換
image_fft = np.fft.fft2(img)
# 預設中心點在左上角，轉換到圖像中心
image_shift = np.fft.fftshift(image_fft)
# image_shift = (image_shift - image_shift.min()) / (image_shift.max() - image_shift.min())
mask = np.zeros(img.shape)
mask_add = mask + np.abs(image_shift).min()
mask[16:36, 41:61] = 1
mask_add[16:36, 41:61] = 0
image_shift = image_shift * mask + mask_add
# image_shift[16:36, 41:61] = np.abs(image_shift).min()
# 將複數轉換成絕對值再取log
image_real = np.log(np.abs(image_shift))
# image_real = np.abs(image_shift)

# 傅立葉逆轉換
image_shift_inverse = np.fft.ifftshift(image_shift)
image_fft_inverse = np.fft.ifft2(image_shift_inverse)
image_real_inverse = np.abs(image_fft_inverse)

plt.subplot(221)
plt.imshow(img, 'gray')
plt.subplot(222)
plt.imshow(image_real, 'gray')
plt.subplot(223)
plt.imshow(image_real_inverse, 'gray')
plt.show()
print(1)
