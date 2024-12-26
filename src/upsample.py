'''
Upsampling methods
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import scipy.io as sio
import time
import os


class upscaler(nn.Module):
    '''
        Upscaler (initial state generator) to convert low-res to high-res initial state
    '''

    def __init__(self):
        super(upscaler, self).__init__()
        self.layers = []
        self.layers.append(
            nn.ConvTranspose2d(2, 8, kernel_size=5, padding=5 // 2, stride=2, output_padding=1, bias=True))
        # self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.Sigmoid())
        self.layers.append(
            nn.ConvTranspose2d(8, 8, kernel_size=5, padding=5 // 2, stride=2, output_padding=1, bias=True))
        # self.layers.append(torch.nn.ReLU())
        # self.layers.append(torch.nn.Sigmoid())
        self.layers.append(nn.Conv2d(8, 2, 1, 1, padding=0, bias=True))
        self.convnet = torch.nn.Sequential(*self.layers)

    def forward(self, h):
        return self.convnet(h)

class upscalerHeat(nn.Module):
    '''
        Upscaler (initial state generator) to convert low-res to high-res initial state
    '''

    def __init__(self):
        super(upscalerHeat, self).__init__()
        self.layers = []
        self.layers.append(
            nn.ConvTranspose2d(1, 8, kernel_size=5, padding=5 // 2, stride=2, output_padding=1, bias=True))
        # self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.Sigmoid())
        self.layers.append(
            nn.ConvTranspose2d(8, 8, kernel_size=5, padding=5 // 2, stride=2, output_padding=1, bias=True))
        # self.layers.append(torch.nn.ReLU())
        # self.layers.append(torch.nn.Sigmoid())
        self.layers.append(nn.Conv2d(8, 1, 1, 1, padding=0, bias=True))
        self.convnet = torch.nn.Sequential(*self.layers)

    def forward(self, h):
        return self.convnet(h)
    

class upscalerBurgers(nn.Module):
    ''' Upscaler (ISG) to convert low-res to high-res initial state '''

    def __init__(self):
        super(upscalerBurgers, self).__init__()
        self.layers = []
        self.up0 = nn.ConvTranspose2d(2, 8, kernel_size=5, padding=5 // 2, stride=2, output_padding=1, bias=True)
        self.tanh = torch.nn.Tanh()
        self.out = nn.Conv2d(8, 2, 1, 1, padding=0, bias=True)
        self.convnet = torch.nn.Sequential(self.up0, self.tanh, self.out)

    def forward(self, h):
        return self.convnet(h)
    