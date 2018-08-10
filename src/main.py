#!/usr/bin/env python

import sys
import time

import torch
import torch.nn as nn

from os.path import join
from processing import data_ingestion, utils
from models import deep_models
from torch import optim
from torchvision import utils as vutils

RESULTS_DIR = './results'
MODELS_DIR = join(RESULTS_DIR, 'models')

def initializer():
    utils.mkdir(RESULTS_DIR)
    utils.mkdir(MODELS_DIR)
    return

def train(train_loader, num_epochs=20):
    autoencoder = deep_models.ConvolutionalDenoisingAutoencoder()
    # weights = torch.randn(2)
    criterion = nn.BCELoss() # binary cross entropy loss
    optimizer = optim.Adadelta(autoencoder.parameters())

    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader, 0):
            ## zero the gradient params
            optimizer.zero_grad()
            ### forward + backprop + optimize
            output = autoencoder(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            print(
                'epoch [{}/{}], loss:{:.4f}'
                .format(epoch+1, num_epochs, loss.data[0])
            )
            # if epoch % 5 == 0:
            #     vutils.save_image(
            #         output.data,
            #         join(RESULTS_DIR, '/image_{}.png'.format(epoch))
            #     )
    torch.save(autoencoder.state_dict(), join(MODELS_DIR, 'autoencoder.pth'))
    return

if __name__ == '__main__':
    initializer()
    train_loader, val_loader = data_ingestion.load_data()
    train(train_loader=train_loader, num_epochs=20)
