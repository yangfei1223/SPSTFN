# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


class BinaryCrossEntropyLoss2D(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyLoss2D, self).__init__()

    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        pos, neg = pred[target == 1], pred[target == 0]
        loss = -0.5 * torch.mean(torch.log(pos)) - 0.5 * torch.mean(torch.log(1-neg))
        return loss


class BooststrapBinaryCrossEntropyLoss2D(nn.Module):
    def __init__(self, k):
        super(BooststrapBinaryCrossEntropyLoss2D, self).__init__()
        self.K = k

    def forward(self, pred, target):
        batch_size = target.size()[0]
        pred = pred.contiguous().view(batch_size, -1)
        target = target.contiguous().view(batch_size, -1)
        loss = 0.0
        for i in range(batch_size):
            pos, neg = pred[i][target[i] == 1], pred[i][target[i] == 0]
            batch_loss = -torch.log(torch.cat((pos, 1-neg)))
            topk_loss, _ = batch_loss.topk(self.K)
            loss += topk_loss.sum() / self.K
        return loss / float(batch_size)




