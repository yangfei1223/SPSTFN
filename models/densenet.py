import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class ImageDenseNet(nn.Module):
    '''
    DenseNet for Image, full resolution
    '''
    def __init__(self, num_channels=3, growth_rate=32, num_layers=14, num_init_features=64, bn_size=4, drop_rate=0):
        super(ImageDenseNet, self).__init__()
        self.model_name = 'DenseNet'
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(num_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        # denseblock, for full resolutuion only one block
        num_features = num_init_features
        block = DenseBlock(num_layers=num_layers, num_input_features=num_features,
                           bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.features.add_module('denseblock', block)
        num_features = num_features + num_layers * growth_rate
        # Final batch normalization
        self.features.add_module('final_norm', nn.BatchNorm2d(num_features))
        self.features.add_module('final_relu', nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.features(x)
        return x


class ImageDenseNet2(nn.Module):
    '''
    2 blocks of DenseNet
    '''
    def __init__(self, num_channels=3, num_out_features=16, num_init_features=32, growth_rate=16, num_layers=(6, 6), bn_size=4, drop_rate=0):
        super(ImageDenseNet2, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(num_channels, num_init_features, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1)
        )   # 1/4
        num_features = num_init_features
        self.block1 = DenseBlock(num_layers=num_layers[0], num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers[0] * growth_rate
        self.norm1 = nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.Conv2d(num_features, num_out_features, 1, bias=False),
            nn.BatchNorm2d(num_out_features),
        )
        # self.trans = Transition(num_input_features=num_features, num_output_features=num_features // 2)
        self.trans = nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features // 2, 1, bias=False)
        )
        num_features = num_features // 2
        self.block2 = DenseBlock(num_layers=num_layers[1], num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers[1] * growth_rate
        self.norm2 = nn.Sequential(
            nn.BatchNorm2d(num_features),
        )

    def forward(self, x):
        x = self.downsample(x)  # 1/4
        x = self.block1(x)   # 1/4
        feat = self.norm1(x)  # 1/4
        x = self.trans(x)
        x = self.block2(x)
        x = self.norm2(x)
        return feat, x
