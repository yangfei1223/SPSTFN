import torch
import torch.nn as nn
import torch.nn.functional as F


class PointUNetQuarter(nn.Module):
    '''
    1/4 DownSample U-Net for Cloud Segmentation
    '''
    def __init__(self, in_channels=11):
        super(PointUNetQuarter, self).__init__()

        self.conv_norm = nn.Sequential(
            nn.Conv2d(12, 4, 1, bias=False),        # multi-scale normal dimension reduce
            nn.BatchNorm2d(4)
        )
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.deconv1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # sampler, +1 to ensure the neg value can be reserve
        x[x != 0] += 1.
        x = F.max_pool2d(x, (7, 3), stride=4, padding=(3, 1))      # 1/4, 64
        x[x != 0] -= 1.
        x, normal = x[:, :7, :, :], x[:, 7:, :, :]
        # pre-process
        x = self.pre(torch.cat((x, self.conv_norm(normal)), dim=1))
        # encoder
        conv1 = self.conv1(x)   # 1/4, 64
        x = F.max_pool2d(conv1, 2, stride=2)
        conv2 = self.conv2(x)   # 1/8, 128
        x = F.max_pool2d(conv2, 2, stride=2)
        conv3 = self.conv3(x)   # 1/16, 256
        x = F.max_pool2d(conv3, 2, stride=2)
        x = self.bottleneck(x)  # 1/32, 512
        # decoder
        x = F.upsample(x, scale_factor=2, mode='bilinear')
        x = self.deconv3(torch.cat((conv3, x), dim=1))
        x = F.upsample(x, scale_factor=2, mode='bilinear')
        x = self.deconv2(torch.cat((conv2, x), dim=1))
        x = F.upsample(x, scale_factor=2, mode='bilinear')
        x = self.deconv1(torch.cat((conv1, x), dim=1))

        return x
