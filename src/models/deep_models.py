
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
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(8, 16, 3),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
            # nn.Conv2d(16, 16, 3),
            # nn.ReLU(True),
            # nn.MaxPool2d(2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, 3),
            nn.ReLU(True),
            # nn.Upsample(size=(2, 2), mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(32, 32, 3),
            nn.ReLU(True),
            # nn.Upsample(size=(2, 2), mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(32, 16, 3),
            nn.ReLU(True),
            # nn.Upsample(size=(2, 2), mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(16, 3, 3),
            nn.ReLU(True),
            # nn.Upsample(size=(2, 2), mode='bilinear', align_corners=True),
            # nn.ConvTranspose2d(16, 8, 3),
            # nn.ReLU(True),
            # nn.Upsample(size=(2, 2), mode='bilinear', align_corners=True),
            # nn.ConvTranspose2d(8, 3, 3),
            nn.Tanh()
        )
        return

    def forward(self, x):
        encoded = self.encoder(x)
        print x.size()
        print
        print encoded.size()
        print
        decoded = self.decoder(encoded)
        print decoded.size()
        print
        return decoded
