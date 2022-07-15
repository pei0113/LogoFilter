import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms


class LOGODatasets(data.Dataset):
    def __init__(self, db_root):
        super(LOGODatasets, self).__init__()
        path_raw = []
        path_db = []
        augmentation_id = []
        for _root, dirs, files in os.walk(os.path.join(db_root, r'DeleteBkg\high')):
            for file in files:
                if file.endswith('.png'):
                    file_raw = file.split('_ffc')[0] + '.png'
                    fn_db = os.path.join(_root, file)
                    fn_raw = os.path.join(_root.replace(r'DeleteBkg\high', r'Raw'), file_raw)
                    for i in range(4):
                        path_db.append(fn_db)
                        path_raw.append(fn_raw)
                        augmentation_id.append(i)
        self.path_raw = path_raw
        self.path_db = path_db
        self.aug_id = augmentation_id
        self.trans = transforms.ToTensor()

    def __getitem__(self, idx):
        path_raw = self.path_raw[idx]
        path_db = self.path_db[idx]

        img_raw = cv2.imread(path_raw, 0)
        img_db = cv2.imread(path_db, 0)

        img_raw = cv2.resize(img_raw, (102, 54))
        img_db = cv2.resize(img_db, (102, 54))

        augmentation_id = self.aug_id
        if augmentation_id == 1:
            img_raw = cv2.flip(img_raw, 0)    # vertical
            img_db = cv2.flip(img_db, 0)
        elif augmentation_id == 2:
            img_raw = cv2.flip(img_raw, 1)    # horizontal
            img_db = cv2.flip(img_db, 1)
        elif augmentation_id == 3:
            img_raw = cv2.flip(img_raw, -1)   # vertical + horizontal
            img_db = cv2.flip(img_db, -1)
        tensor_raw = self.trans(Image.fromarray(img_raw))
        tensor_db = self.trans(Image.fromarray(img_db))

        # check plot
        # plt.subplot(121)
        # plt.imshow(img_raw, 'gray', vmin=0, vmax=255)
        # plt.subplot(122)
        # plt.imshow(img_db, 'gray', vmin=0, vmax=255)
        # plt.show()

        out = {
            'path_input': path_raw,
            'path_output': path_db,
            'tensor_input': tensor_raw,
            'tensor_output': tensor_db
        }
        return out

    def __len__(self):
        return len(self.path_db)


if __name__ == "__main__":
    datasets = LOGODatasets(r'C:\Users\pei.liu\Downloads\Data_510')

    for nth_batch, inputs in enumerate(datasets):
        tensor_input = inputs['tensor_input']
        tensor_output = inputs['tensor_output']
