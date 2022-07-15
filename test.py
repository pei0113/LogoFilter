import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters.rank import equalize
from skimage.morphology import disk

import torch
import torch.utils.data as data

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

# load data
train_datasets = LOGODatasets(r'C:\Users\pei.liu\Downloads\Data_510_FFC_BeforeAfter')
nn_split = int(len(train_datasets)*0.9)
train_datasets, valid_datasets = data.random_split(train_datasets, [nn_split, len(train_datasets)-nn_split])
train_loader = data.DataLoader(train_datasets, batch_size=_batch_size, shuffle=True, num_workers=0)
valid_loader = data.DataLoader(valid_datasets, batch_size=_batch_size, shuffle=False)
print("[*] Train data: {}, Test data: {}".format(len(train_datasets), len(valid_datasets)))

# load model
model = LogoFilter().to('cuda')
kpt_path = 'model/epoch_45_trainLoss=0.0715_validLoss=0.007537.pth'
model.load_state_dict(torch.load(kpt_path))

### TEST MODE
selem = disk(20)
model.eval()
for nth_batch, inputs in enumerate(valid_loader):
    tensor_input = inputs['tensor_input'].cuda()
    tensor_output = inputs['tensor_output'].cuda()
    path = inputs['path_input'][0]

    img_out = model(tensor_input)
    img_out = img_out.cpu().data.numpy()[0, 0, :, :]*255

    img_input = tensor_input.cpu().data.numpy()[0, 0, :, :]*255
    img_gt = tensor_output.cpu().data.numpy()[0, 0, :, :]*255

    eq_input = equalize(img_input.astype('uint8'), selem=selem)
    eq_gt = equalize(img_gt.astype('uint8'), selem=selem)
    eq_output = equalize(img_out.astype('uint8'), selem=selem)

    plt.subplot(241), plt.title('input')
    plt.imshow(img_input, 'gray', vmin=0, vmax=255)
    plt.subplot(242), plt.title('gt')
    plt.imshow(img_gt, 'gray', vmin=0, vmax=255)
    plt.subplot(243), plt.title('img_out')
    plt.imshow(img_out, 'gray', vmin=0, vmax=255)
    plt.subplot(244), plt.title('img_out')
    plt.imshow(img_input-img_out, 'gray')
    plt.subplot(245), plt.title('input_eq')
    plt.imshow(eq_input, 'gray', vmin=0, vmax=255)
    plt.subplot(246), plt.title('gt_eq')
    plt.imshow(eq_gt, 'gray', vmin=0, vmax=255)
    plt.subplot(247), plt.title('out_eq')
    plt.imshow(eq_output, 'gray', vmin=0, vmax=255)
    plt.subplot(248), plt.title('out_eq')
    plt.imshow(eq_input-eq_output, 'gray', vmin=0, vmax=255)

    # plt.savefig(path.replace('Raw\\', 'Prediction\\'))
    plt.show()
