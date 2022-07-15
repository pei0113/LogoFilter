import torch.nn as nn
import torch.nn.functional as F


# v0
# class LogoFilter(nn.Module):
#     def __init__(self):
#         super(LogoFilter, self).__init__()
#         self.conv2d_1 = nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False)
#         self.conv2d_2 = nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False)
#         self.conv2d_3 = nn.Conv2d(8, 4, kernel_size=3, padding=1, bias=False)
#         self.conv2d_4 = nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=False)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x = self.relu(self.conv2d_1(x))
#         x = self.relu(self.conv2d_2(x))
#         x = self.relu(self.conv2d_3(x))
#         x = self.sigmoid(self.conv2d_4(x))
#         return x
#v1
class LogoFilter(nn.Module):
    def __init__(self):
        super(LogoFilter, self).__init__()
        self.conv2d_1 = nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False)
        self.conv2d_2 = nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False)
        self.conv2d_3 = nn.Conv2d(8, 4, kernel_size=3, padding=1, bias=False)
        self.conv2d_4 = nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(4)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(4)
        self.bn4 = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout2d(0.2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv2d_1(x)))
        # x = self.drop(x)
        x = self.relu(self.bn2(self.conv2d_2(x)))
        # x = self.drop(x)
        x = self.relu(self.bn3(self.conv2d_3(x)))
        # x = self.drop(x)
        x = self.sigmoid(self.bn4(self.conv2d_4(x)))
        return x


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.conv2d_0 = nn.Conv2d(1, 6, kernel_size=3, padding=1, bias=False)
        self.conv2d_1 = nn.Conv2d(6, 16, kernel_size=3, padding=1, bias=False)
        self.conv2d_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)
        self.conv2d_3 = nn.Conv2d(16, 6, kernel_size=3, padding=1, bias=False)
        self.conv2d_4 = nn.Conv2d(6, 1, kernel_size=1, bias=False)
        self.deconv2d_0 = nn.ConvTranspose2d(16, 6, kernel_size=2, stride=2, bias=False)
        self.max_pool = nn.MaxPool2d(2, return_indices=True)
        self.max_unpool = nn.MaxUnpool2d(2)
        # self.up_sample0 = upsample(16, padX=False, padY=True, bilinear=False)
        # self.up_sample1 = upsample(6, padX=False, padY=True, bilinear=False)
        self.up_sample0 = upsample(16, padX=True, padY=True, bilinear=False)
        self.up_sample1 = upsample(6, padX=False, padY=False, bilinear=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.conv2d_0(x))               # (32, 71, 1) -> (32, 71, 6)
        x, indices1 = self.max_pool(x)                # (31, 71, 6) -> (16, 35, 6)
        x = self.relu(self.conv2d_1(x))               # (16, 35, 6) -> (16, 35, 16)
        x, indices2 = self.max_pool(x)                # (16, 35, 16) -> (8, 17, 16)
        x = self.relu(self.conv2d_2(x))               # (8, 17, 16) -> (8, 17, 16)
        x = self.up_sample0(x)                        # (8, 17, 16) -> (16, 35, 16)
        x = self.relu(self.conv2d_3(x))               # (16, 35, 16) -> (16, 35, 6)
        x = self.up_sample1(x)                        # (16, 35, 6) -> (32, 71, 6)
        x = self.sigmoid(self.conv2d_4(x))            # (32, 71, 6) -> (32, 71, 1)
        return x


class upsample(nn.Module):
    def __init__(self, _channel, padX, padY, bilinear=True):
        super(upsample, self).__init__()
        self.padY = padY
        self.padX = padX

        if bilinear:
            self.up = nn.UpsamplingNearest2d(scale_factor=2)
        else:
            self.up = nn.ConvTranspose2d(_channel, _channel, kernel_size=2, stride=2, bias=False)
    def forward(self, x):
        x = self.up(x)
        if self.padX and self.padY:
            x = F.pad(x, (0, 1, 0, 1))
        else:
            if self.padY:
                x = F.pad(x, (0, 1, 0, 0))
            if self.padX:
                x = F.pad(x, (0, 0, 0, 1))

        return x