import time
import random
import numpy as np

import torch
torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn as nn
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
Epochs = 50
_batch_size = 8
set_random_seed(2022)

# load data
train_datasets = LOGODatasets(r'C:\Users\pei.liu\Downloads\Data_510_FFC_BeforeAfter')
nn_split = int(len(train_datasets)*0.9)
train_datasets, valid_datasets = data.random_split(train_datasets, [nn_split, len(train_datasets)-nn_split])
train_loader = data.DataLoader(train_datasets, batch_size=_batch_size, shuffle=True, num_workers=0)
valid_loader = data.DataLoader(valid_datasets, batch_size=_batch_size, shuffle=False)
print("[*] Train data: {}, Test data: {}".format(len(train_datasets), len(valid_datasets)))

# load model
model = LogoFilter().to(device)

# train
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(Epochs):
    t_start = time.time()
    ### TRAIN MODE
    epoch_total_loss = 0
    model.train()
    for nth_batch, inputs in enumerate(train_loader):
        tensor_input = inputs['tensor_input'].to('cuda')
        tensor_output = inputs['tensor_output'].to('cuda')

        img_generate = model(tensor_input)

        loss = criterion(img_generate, tensor_output)
        epoch_total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_train_loss = epoch_total_loss / len(train_loader)

    ### EVALUATE MODE
    epoch_total_loss = 0
    model.eval()
    with torch.no_grad():
        for nth_batch, inputs in enumerate(valid_loader):
            tensor_input = inputs['tensor_input'].cuda()
            tensor_output = inputs['tensor_output'].cuda()

            img_generate = model(tensor_input)
            loss = criterion(img_generate, tensor_output)
            epoch_total_loss += loss

        avg_valid_loss = epoch_total_loss / len(train_loader)

    print("[*] Time: {:.1f} || Epoch {} / {} || [[Train]] total:{:.5f}, [[Valid]] total:{:.5f}".format(
        time.time() - t_start,
        epoch + 1, Epochs, avg_train_loss, avg_valid_loss))

    # save model
    torch.save(model.state_dict(),
               'model\\epoch_{}_trainLoss={:.4f}_validLoss={:.6f}.pth'.format(epoch + 1, avg_train_loss, avg_valid_loss), _use_new_zipfile_serialization=False)
