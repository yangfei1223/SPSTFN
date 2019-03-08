# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from pytorch_spn.modules.gaterecurrent2dnoind import GateRecurrent2dnoind


class SpatialPropagationBlock(nn.Module):
    '''
    SPN module
    '''
    def __init__(self, n_features=32):
        super(SpatialPropagationBlock, self).__init__()
        self.n_features = n_features
        # propagation layers
        self.Propagator_x1 = GateRecurrent2dnoind(True, False)
        self.Propagator_x2 = GateRecurrent2dnoind(True, True)
        self.Propagator_y1 = GateRecurrent2dnoind(False, False)
        self.Propagator_y2 = GateRecurrent2dnoind(False, True)

    def normalize_gate(self, g1, g2, g3):
        sum_abs = g1.abs() + g2.abs() + g3.abs() + 1e-7
        mask_need_norm = sum_abs.ge(1)
        mask_need_norm = mask_need_norm.float()
        g1_norm = torch.div(g1, sum_abs)
        g2_norm = torch.div(g2, sum_abs)
        g3_norm = torch.div(g3, sum_abs)
        g1 = torch.add(-mask_need_norm, 1) * g1 + mask_need_norm * g1_norm
        g2 = torch.add(-mask_need_norm, 1) * g2 + mask_need_norm * g2_norm
        g3 = torch.add(-mask_need_norm, 1) * g3 + mask_need_norm * g3_norm
        return g1, g2, g3

    def forward(self, x, mask):
        # left->right
        x1_g1 = x[:, 0:self.n_features, :, :]
        x1_g2 = x[:, self.n_features:2*self.n_features, :, :]
        x1_g3 = x[:, 2*self.n_features:3*self.n_features, :, :]
        x1_g1, x1_g2, x1_g3 = self.normalize_gate(x1_g1, x1_g2, x1_g3)
        # right->left
        x2_g1 = x[:, 3*self.n_features:4*self.n_features, :, :]
        x2_g2 = x[:, 4*self.n_features:5*self.n_features, :, :]
        x2_g3 = x[:, 5*self.n_features:6*self.n_features, :, :]
        x2_g1, x2_g2, x2_g3 = self.normalize_gate(x2_g1, x2_g2, x2_g3)
        # up->bottom
        y1_g1 = x[:, 6*self.n_features:7*self.n_features, :, :]
        y1_g2 = x[:, 7*self.n_features:8*self.n_features, :, :]
        y1_g3 = x[:, 8*self.n_features:9*self.n_features, :, :]
        y1_g1, y1_g2, y1_g3 = self.normalize_gate(y1_g1, y1_g2, y1_g3)
        # bottom->up
        y2_g1 = x[:, 9*self.n_features:10*self.n_features, :, :]
        y2_g2 = x[:, 10*self.n_features:11*self.n_features, :, :]
        y2_g3 = x[:, 11*self.n_features:12*self.n_features, :, :]
        y2_g1, y2_g2, y2_g3 = self.normalize_gate(y2_g1, y2_g2, y2_g3)

        # propagate
        x1 = self.Propagator_x1.forward(mask, x1_g1, x1_g2, x1_g3)
        x2 = self.Propagator_x2.forward(mask, x2_g1, x2_g2, x2_g3)
        y1 = self.Propagator_y1.forward(mask, y1_g1, y1_g2, y1_g3)
        y2 = self.Propagator_y2.forward(mask, y2_g1, y2_g2, y2_g3)
        mask = torch.max(torch.max(x1, x2), torch.max(y1, y2))

        return mask

