import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms

from datasets import LOGODatasets
from network import LogoFilter, AutoEncoder


def set_random_seed(n):
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    torch.cuda.manual_seed(n)
    torch.backends.cudnn.deterministic=True
    return print('[*] Set random seed: {}'.format(n))

# init
_batch_size = 1
set_random_seed(2022)

# load model
model = LogoFilter().to('cuda')
kpt_path = 'model/epoch_50_trainLoss=0.0030_validLoss=0.000376.pth'
model.load_state_dict(torch.load(kpt_path))

### TEST MODE
model.eval()

# load data
img_paths = []
db_root = r'C:\Users\pei.liu\Downloads\Data_510\Raw'
# db_root = r'D:\DB\ET5\logo\LogoImage_20220617\block\logoStrong\GoodImg\Raw'
for _root, dirs, files in os.walk(db_root):
    for file in files:
        if file.endswith('png'):
            img_paths.append(os.path.join(_root, file))

trans = transforms.ToTensor()
for nth_img, img_path in enumerate(img_paths):
    img = cv2.imread(img_path, 0)

    img_int16 = np.int16(img)
    img_highpass = img_int16 - cv2.blur(img_int16, (3, 3))
    img_lowpass = cv2.blur(img_int16, (3, 3))
    var = np.mean(np.abs(img_highpass))
    print("RV:{}".format(var))
    plt.subplot(221)
    plt.imshow(img, 'gray')
    plt.subplot(222)
    plt.imshow(img_highpass, 'gray')
    plt.subplot(223)
    plt.imshow(img_lowpass, 'gray')
    plt.show()

    img_resize = cv2.resize(img, (102, 54))
    # img_resize = cv2.flip(img_resize, 1)
    img_tensor = trans(Image.fromarray(img_resize)).cuda()
    img_tensor = torch.unsqueeze(img_tensor, 0)
    img_out = model(img_tensor)
    img_out = img_out.cpu().data.numpy()[0, 0, :, :]
    plt.subplot(121)#, plt.title('input')
    plt.imshow(img_tensor.cpu().data.numpy()[0, 0, :, :], 'gray', vmin=0, vmax=1)
    # plt.subplot(132), plt.title('gt')
    # plt.imshow(tensor_output.cpu().data.numpy()[0, 0, :, :], 'gray', vmin=0, vmax=1)
    plt.subplot(122)#, plt.title('img_out')
    plt.imshow(img_out, 'gray', vmin=0, vmax=1)
    plt.show()
