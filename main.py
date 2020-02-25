# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 17:46:11 2020

@author: David
"""

import os
import torch
import torch.nn as nn
import numpy as np
import time
import torch.optim as optim
import matplotlib.pyplot as plt
from dataPrep import add_images_to_dataset
from dataLoad import load_data_sets, numpy_to_pytorch
from pyTorchTools import Flatten, train, init_weight, check_accuracy, error_analysis, build_pytorch_model
from numpyModelNN import *
from numpySolver import *
from numpyLayers import *

imPathOrig = os.getcwd() + "/newImages/"
imPathPrep = os.getcwd() + "/imagesPrepared/"

imH, imW = 64, 64
areNewImagesAvailable = False
if areNewImagesAvailable:
    add_images_to_dataset(imPathOrig,imPathPrep,imSize=(imH,imW),save=True,flips=True)

data, meanIm, stdIm = load_data_sets(imPathPrep, split=(80,10,10), imSize=(imH,imW)) # dictionary
yTrain, yDev, yTest = data['Y_train'], data['Y_dev'], data['Y_test']
y = np.concatenate((yTrain, yDev, yTest), axis=0)
for i in range(6):
    print(np.sum(y == i) / y.shape[0])
    print(np.sum(y == i))


usePyTorch = False
if usePyTorch:
    batch_size = 256
    loaders = numpy_to_pytorch(data, batch_size)
    del data
    build_pytorch_model()
    
    print_every = 26
    learning_rate = 3e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 15

    tic = time.time()
    losses, iters, trainAccs, devAccs = train(model, loaders, optimizer, epochs, print_every)
    trainingTime = time.time() - tic
    print("Training time was: ", trainingTime, " seconds")
    
    performErrorAnalysis = False
    results_pytorch(iters,losses,trainAccs,devAccs,loaders,model,stdIm,meanIm,performErrorAnalysis)

    #pathStateDict = os.getcwd() + "/20200223_1149.pt"
    #torch.save(model.state_dict(), pathStateDict)
elif usePyTorch is False:
    num_train = 512
    small_data = {
      'X_train': data['X_train'][:num_train],
      'Y_train': data['Y_train'][:num_train],
      'X_val': data['X_dev'],
      'Y_val': data['Y_dev'],
    }
    
    channel_in = 3
    channel_1 = 16
    channel_2 = 16
    channel_3 = 32
    channel_4 = 32
    channel_5 = 64
    channel_6 = 64
    channel_7 = 128
    channel_8 = 128
    architecture = [
        ('Conv2D', (channel_1, channel_in, 3, 3, {'pad': 1, 'stride': 1})),
        ('Batchnorm2D', (channel_1)),
        ('ReLU',()),
        ('Conv2D', (channel_2, channel_1, 3, 3, {'pad': 1, 'stride': 1})),
        ('Batchnorm2D', (channel_2)),
        ('ReLU',()),
        ('Maxpool2D',(2,2,2)), # (pool_height, pool_width, stride)
        ####
        ('Conv2D', (channel_3, channel_2, 3, 3, {'pad': 1, 'stride': 1})),
        ('Batchnorm2D', (channel_3)),
        ('ReLU',()),
        ('Conv2D', (channel_4, channel_3, 3, 3, {'pad': 1, 'stride': 1})),
        ('Batchnorm2D', (channel_4)),
        ('ReLU',()),
        ('Maxpool2D',(2,2,2)), # (pool_height, pool_width, stride)
        ####
        ('Conv2D', (channel_5, channel_4, 3, 3, {'pad': 1, 'stride': 1})),
        ('Batchnorm2D', (channel_5)),
        ('ReLU',()),
        ('Conv2D', (channel_6, channel_5, 3, 3, {'pad': 1, 'stride': 1})),
        ('Batchnorm2D', (channel_6)),
        ('ReLU',()),
        ('Maxpool2D',(2,2,2)), # (pool_height, pool_width, stride)
        ####
        ('Conv2D', (channel_7, channel_6, 3, 3, {'pad': 1, 'stride': 1})),
        ('Batchnorm2D', (channel_7)),
        ('ReLU',()),
        ('Conv2D', (channel_8, channel_7, 3, 3, {'pad': 1, 'stride': 1})),
        ('Batchnorm2D', (channel_8)),
        ('ReLU',()),
        ####
        ('Affine', (int(channel_8*imW*imH/(4*4*4)), 16)),
        ('Batchnorm1D', (16)),
        ('ReLU',()),
        ('Dropout',(0.9)),
        ('Affine', (16,6))
    ]
    
    learning_rate = 1e-3
    model = NeuralNet(architecture, input_dim = 3*64*64, num_classes = 6,
                  dtype=np.float64)
    
    solver = Solver(model, small_data,
                    print_every=1, num_epochs=10, batch_size=32,
                    update_rule='adam',
                    optim_config={
                      'learning_rate': learning_rate,
                    }
             )
    solver.train()
