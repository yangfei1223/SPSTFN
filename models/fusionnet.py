# -*- coding:utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F
from models import *


class Baseline(nn.Module):
    '''
    Baseline of the proposed method
    '''

    def __init__(self, n_x_features=512, n_mask_features=64, propagation_features=32, n_classes=2):
        super(Baseline, self).__init__()
        self.model_name = 'FusionBaseline'
        # feature extract
        self.features = PointUNetQuarter()

        # convolution classifiers
        self.conv_classifier = nn.Sequential(
            nn.Conv2d(n_mask_features, 1, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Sigmoid()
        )

    def forward(self, cloud):
        x = self.features(cloud)
        x = self.conv_classifier(x)
        return x


class FusionDenseNetSP(nn.Module):
    '''
       propagation only in perspective view
    '''

    def __init__(self, n_x_features=512, n_mask_features=64, propagation_features=32, n_classes=2):
        super(FusionDenseNetSP, self).__init__()
        self.model_name = 'FusionDenseNetSP'
        # feature extract
        self.features = ImageDenseNet()
        self.mask = PointUNetQuarter()

        # if the batch normalization is needed ? This is a problem to be explored.
        self.x_perspective_bottleneck = nn.Sequential(
            # nn.Conv2d(n_x_features, propagation_features * 12, 1),
            nn.Conv2d(n_x_features, propagation_features * 12, 3, padding=1),
            # nn.BatchNorm2d(propagation_features * 12)
        )

        self.mask_perspective_bottleneck = nn.Sequential(
            # nn.Conv2d(n_mask_features, propagation_features, 1),
            nn.Conv2d(n_mask_features, propagation_features, 3, padding=1),
            # nn.BatchNorm2d(propagation_features)
        )

        # spatial propagation
        self.spn_perspective = SpatialPropagationBlock(n_features=32)

        # refine convolution after concatenate
        self.last_conv = nn.Sequential(
            nn.Conv2d(propagation_features, n_mask_features, 3, padding=1, bias=False),
            nn.BatchNorm2d(n_mask_features),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(),
            nn.Conv2d(n_mask_features, n_mask_features, 3, padding=1, bias=False),
            nn.BatchNorm2d(n_mask_features),
            nn.ReLU(inplace=True),
            # nn.Dropout2d()
        )

        # convolution classifiers
        self.coarse_classifier = nn.Sequential(
            nn.Conv2d(n_mask_features, 1, kernel_size=3, padding=1),
            # nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Sigmoid()
        )

        self.fine_classifier = nn.Sequential(
            nn.Conv2d(n_mask_features, 1, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Sigmoid()
        )

    def forward(self, im, cloud):
        x = self.features(im)
        mask = self.mask(cloud)  # 1/4
        coarse = self.coarse_classifier(mask)

        # fusion
        mask = self.spn_perspective(self.x_perspective_bottleneck(x), self.mask_perspective_bottleneck(mask))
        mask = self.last_conv(mask)

        fine = self.fine_classifier(mask)

        return coarse, fine


class FusionDenseNet(nn.Module):
    '''
       FusionDenseNet ver.1
       '''

    def __init__(self, n_x_features=512, n_mask_features=64, propagation_features=32, n_classes=2):
        super(FusionDenseNet, self).__init__()
        self.model_name = 'FusionDenseNet'
        # feature extract
        self.features = ImageDenseNet()
        self.mask = PointUNetQuarter()

        # if the batch normalization is needed ? This is a problem to be explored.
        self.x_perspective_bottleneck = nn.Sequential(
            # nn.Conv2d(n_x_features, propagation_features * 12, 1),
            nn.Conv2d(n_x_features, propagation_features * 12, 3, padding=1),
            # nn.BatchNorm2d(propagation_features * 12)
        )
        self.x_bev_bottleneck = nn.Sequential(
            # nn.Conv2d(n_x_features, propagation_features * 12, 1),
            nn.Conv2d(n_x_features, propagation_features * 12, 3, padding=1),
            # nn.BatchNorm2d(propagation_features * 12)
        )
        self.mask_perspective_bottleneck = nn.Sequential(
            # nn.Conv2d(n_mask_features, propagation_features, 1),
            nn.Conv2d(n_mask_features, propagation_features, 3, padding=1),
            # nn.BatchNorm2d(propagation_features)
        )
        self.mask_bev_bottleneck = nn.Sequential(
            # nn.Conv2d(n_mask_features, propagation_features, 1),
            nn.Conv2d(n_mask_features, propagation_features, 3, padding=1),
            # nn.BatchNorm2d(propagation_features)
        )

        # spatial transform
        self.stn_x = SpatialTransformBlock()
        self.stn_mask = SpatialTransformBlock()
        self.stn_back = SpatialTransformBlock(inverse=True)
        # spatial propagation
        self.spn_perspective = SpatialPropagationBlock(n_features=32)
        self.spn_bev = SpatialPropagationBlock(n_features=32)

        # refine convolution after concatenate
        self.last_conv = nn.Sequential(
            nn.Conv2d(propagation_features * 2, n_mask_features, 3, padding=1, bias=False),
            nn.BatchNorm2d(n_mask_features),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(),
            nn.Conv2d(n_mask_features, n_mask_features, 3, padding=1, bias=False),
            nn.BatchNorm2d(n_mask_features),
            nn.ReLU(inplace=True),
            # nn.Dropout2d()
        )

        # convolution classifiers
        self.coarse_classifier = nn.Sequential(
            nn.Conv2d(n_mask_features, 1, kernel_size=3, padding=1),
            # nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Sigmoid()
        )

        self.fine_classifier = nn.Sequential(
            nn.Conv2d(n_mask_features, 1, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Sigmoid()
        )

    def forward(self, im, cloud, theta, shift):
        x = self.features(im)
        mask = self.mask(cloud)  # 1/4
        coarse = self.coarse_classifier(mask)

        # fusion
        mask_perspective = self.spn_perspective(self.x_perspective_bottleneck(x),
                                                self.mask_perspective_bottleneck(mask))
        x_bev = self.stn_x(x, theta, shift, torch.Size((x.shape[0], x.shape[1], 200, 100)))
        mask_bev = self.stn_mask(mask, theta, shift, torch.Size((mask.shape[0], mask.shape[1], 200, 100)))
        mask_bev = self.spn_bev(self.x_bev_bottleneck(x_bev), self.mask_bev_bottleneck(mask_bev))
        mask_bev = self.stn_back(mask_bev, theta, shift, torch.Size((mask_bev.shape[0], mask_bev.shape[1], 72, 304)))
        mask = torch.cat((mask_perspective, mask_bev), dim=1)
        mask = self.last_conv(mask)

        fine = self.fine_classifier(mask)

        return coarse, fine


class FusionDenseNetBev(nn.Module):
    '''
    FusionDenseNet output bird view
    '''

    def __init__(self, n_x_features=512, n_mask_features=64, propagation_features=32, n_classes=2):
        super(FusionDenseNetBev, self).__init__()
        self.model_name = 'FusionDenseNet'
        # feature extract
        self.features = ImageDenseNet()
        self.mask = PointUNetQuarter()

        # if the batch normalization is needed ? This is a problem to be explored.
        self.x_perspective_bottleneck = nn.Sequential(
            # nn.Conv2d(n_x_features, propagation_features * 12, 1),
            nn.Conv2d(n_x_features, propagation_features * 12, 3, padding=1),
            # nn.BatchNorm2d(propagation_features * 12)
        )
        self.x_bev_bottleneck = nn.Sequential(
            # nn.Conv2d(n_x_features, propagation_features * 12, 1),
            nn.Conv2d(n_x_features, propagation_features * 12, 3, padding=1),
            # nn.BatchNorm2d(propagation_features * 12)
        )
        self.mask_perspective_bottleneck = nn.Sequential(
            # nn.Conv2d(n_mask_features, propagation_features, 1),
            nn.Conv2d(n_mask_features, propagation_features, 3, padding=1),
            # nn.BatchNorm2d(propagation_features)
        )
        self.mask_bev_bottleneck = nn.Sequential(
            # nn.Conv2d(n_mask_features, propagation_features, 1),
            nn.Conv2d(n_mask_features, propagation_features, 3, padding=1),
            # nn.BatchNorm2d(propagation_features)
        )

        # spatial transform
        self.stn_x = SpatialTransformBlock()
        self.stn_mask = SpatialTransformBlock()
        self.stn_perspective = SpatialTransformBlock()
        # spatial propagation
        self.spn_perspective = SpatialPropagationBlock(n_features=32)
        self.spn_bev = SpatialPropagationBlock(n_features=32)

        # refine convolution after concatenate
        self.last_conv = nn.Sequential(
            nn.Conv2d(propagation_features * 2, n_mask_features, 3, padding=1, bias=False),
            nn.BatchNorm2d(n_mask_features),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(),
            nn.Conv2d(n_mask_features, n_mask_features, 3, padding=1, bias=False),
            nn.BatchNorm2d(n_mask_features),
            nn.ReLU(inplace=True),
            # nn.Dropout2d()
        )

        # convolution classifiers
        self.coarse_classifier = nn.Sequential(
            nn.Conv2d(n_mask_features, 1, kernel_size=3, padding=1),
            # nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Sigmoid()
        )

        self.fine_classifier = nn.Sequential(
            nn.Conv2d(n_mask_features, 1, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Sigmoid()
        )

    def forward(self, im, cloud, theta, shift):
        x = self.features(im)
        mask = self.mask(cloud)  # 1/4
        coarse = self.coarse_classifier(mask)

        # fusion
        mask_perspective = self.spn_perspective(self.x_perspective_bottleneck(x),
                                                self.mask_perspective_bottleneck(mask))
        mask_perspective = self.stn_perspective(mask_perspective, theta, shift,
                                                torch.Size((mask_perspective.shape[0], mask_perspective.shape[1], 200, 100)))

        x_bev = self.stn_x(x, theta, shift, torch.Size((x.shape[0], x.shape[1], 200, 100)))
        mask_bev = self.stn_mask(mask, theta, shift, torch.Size((mask.shape[0], mask.shape[1], 200, 100)))
        mask_bev = self.spn_bev(self.x_bev_bottleneck(x_bev), self.mask_bev_bottleneck(mask_bev))

        mask = torch.cat((mask_perspective, mask_bev), dim=1)
        mask = self.last_conv(mask)

        fine = self.fine_classifier(mask)

        return coarse, fine


