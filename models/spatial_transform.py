# -*- coding:utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable


class BevGridGenerator(Function):
    @staticmethod
    def forward(ctx, theta, shift, size):
        N, C, H, W = size
        rate = 800./H
        grid_res = 0.05*rate
        base_grid = theta.new(N, H, W, 3)
        linear_points = torch.arange(-10 + grid_res / 2, 10, grid_res)
        base_grid[:, :, :, 0] = torch.ger(torch.ones(H), linear_points).expand_as(base_grid[:, :, :, 0])
        linear_points = torch.arange(46 - grid_res / 2, 6, -grid_res)
        base_grid[:, :, :, 1] = torch.ger(linear_points, torch.ones(W)).expand_as(base_grid[:, :, :, 1])
        base_grid[:, :, :, 2] = 1
        grid = torch.bmm(base_grid.view(N, H * W, 3), theta.permute(0, 2, 1))
        grid = grid.view(N, H, W, 3)
        grid_z = torch.unsqueeze(grid[:, :, :, 2], dim=3)
        grid = grid / grid_z
        grid[:, :, :, 0] = (grid[:, :, :, 0]) / 608 - 1
        grid[:, :, :, 1] = (grid[:, :, :, 1] - shift.view(N, 1, 1)) / 144 - 1
        return grid[:, :, :, :2]

    @staticmethod
    def backward(ctx, grad_grid):
        return None, None, None


class PerspectiveGridGenerator(Function):
    @staticmethod
    def forward(ctx, theta, shift, size):
        N, C, H, W = size
        rate = 288./H
        theta_inverse = torch.zeros_like(theta)
        for i in range(N):
            theta_inverse[i, :, :] = torch.inverse(theta[i, :, :])
        base_grid = theta.new(N, H, W, 3)
        linear_points = torch.arange(0, 1216, rate)
        base_grid[:, :, :, 0] = torch.ger(torch.ones(H), linear_points).expand_as(base_grid[:, :, :, 0])
        linear_points = torch.arange(0, 288, rate)
        base_grid[:, :, :, 1] = torch.ger(linear_points, torch.ones(W)).expand_as(base_grid[:, :, :, 1]) + shift.view(N, 1, 1).cpu()
        base_grid[:, :, :, 2] = 1
        grid = torch.bmm(base_grid.view(N, H * W, 3), theta_inverse.permute(0, 2, 1))
        grid = grid.view(N, H, W, 3)
        grid_z = torch.unsqueeze(grid[:, :, :, 2], dim=3)
        grid = grid / grid_z
        grid[:, :, :, 0] = ((grid[:, :, :, 0] + 10) / 0.05) / 200 - 1
        grid[:, :, :, 1] = (800 - (grid[:, :, :, 1] - 6) / 0.05) / 400 - 1
        return grid[:, :, :, :2]

    @staticmethod
    def backward(ctx, grad_grid):
        return None, None, None


class SpatialTransformBlock(nn.Module):
    '''
    Spatial Transform Block
    '''
    def __init__(self, inverse=False):
        '''
        Spatial Transform module
        :param inverse: bool, False for transform to bird view, True otherwise
        '''
        super(SpatialTransformBlock, self).__init__()
        self.grid_generator = self.perspective_grid_generator if inverse else self.bev_grid_generator

    def bev_grid_generator(self, theta, shift, size):
        return BevGridGenerator.apply(theta, shift, size)

    def perspective_grid_generator(self, theta, shift, size):
        return PerspectiveGridGenerator.apply(theta, shift, size)

    def forward(self, x, theta, shift, size):
        grid = self.grid_generator(theta, shift, size)
        x = F.grid_sample(x, grid, padding_mode='border')
        return x

