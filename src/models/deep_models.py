
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F

import numpy as np
import pandas as pd

class ConvolutionalDenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(
            ConvolutionalDenoisingAutoencoder,
            self).__init__()
        # ENCODER
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 16, 3)
        # DECODER
        self.conv4 = nn.Conv2d(16, 32, 3)
        self.upsample1 = nn.Upsample( # equivalent of upsample2d
            size=(2, 2),
            mode='bilinear',
            align_corners=True)
        self.conv5 = nn.Conv2d(32, 32, 3)
        self.upsample2 = nn.Upsample( # equivalent of upsample2d
            size=(2, 2),
            mode='bilinear',
            align_corners=True)
        self.conv6 = nn.Conv2d(32, 16, 3)
        self.upsample3 = nn.Upsample( # equivalent of upsample2d
            size=(2, 2),
            mode='bilinear',
            align_corners=True)
        self.conv7 = nn.Conv2d(16, 16, 3)
        self.upsample4 = nn.Upsample( # equivalent of upsample2d
            size=(2, 2),
            mode='bilinear',
            align_corners=True)
        self.conv8 = nn.Conv2d(16, 8, 3)
        self.upsample5 = nn.Upsample( # equivalent of upsample2d
            size=(2, 2),
            mode='bilinear',
            align_corners=True)
        self.conv9 = nn.Conv2d(8, 3, 3)
        return

    def forward(self, x):
        ## ENCODER
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        encoded = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        ## DECODER
        x = self.upsample1(F.relu(self.conv4(x)))
        x = self.upsample2(F.relu(self.conv5(x)))
        x = self.upsample3(F.relu(self.conv6(x)))
        x = self.upsample4(F.relu(self.conv7(x)))
        x = self.upsample5(F.relu(self.conv8(x)))
        decoded = self.conv9(x)
        return decoded
