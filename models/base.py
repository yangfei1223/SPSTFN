# -*- coding:utf-8 -*-

import torch
import time


class Base(torch.nn.Module):
    '''
    Foundation class
    '''
    def __init__(self):
        super(Base, self).__init__()
        self.model_name = 'Base'

    def load(self, filename):
        self.load_state_dict(torch.load(filename))

    def save(self, filename=None):
        if filename is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            filename = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), filename)
        return filename
