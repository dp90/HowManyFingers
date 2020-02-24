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
from pyTorchTools import Flatten, train, init_weight, check_accuracy, error_analysis

imPathOrig = os.getcwd() + "/newImages/"
imPathPrep = os.getcwd() + "/imagesPrepared/"

imH, imW = 64, 64
areNewImagesAvailable = False
if areNewImagesAvailable:
    add_images_to_dataset(imPathOrig,imPathPrep,imSize=(imH,imW),save=True,flips=True)

usePyTorch = True
data, meanIm, stdIm = load_data_sets(imPathPrep, split=(80,10,10), imSize=(imH,imW)) # dictionary
if usePyTorch:
    batch_size = 256
    loaders = numpy_to_pytorch(data, batch_size)
    del data

"""
yTrain, yDev, yTest = data['Y_train'], data['Y_dev'], data['Y_test']
y = np.concatenate((yTrain, yDev, yTest), axis=0)
for i in range(6):
    print(np.sum(y == i) / y.shape[0])
    print(np.sum(y == i))
"""

print_every = 26
in_channel = 3
channel_1 = 16
channel_2 = 16
channel_3 = 32
channel_4 = 32
channel_5 = 64
channel_6 = 64
channel_7 = 128
channel_8 = 128
learning_rate = 3e-3

pDropOut = 0.75
model = nn.Sequential(
    nn.Conv2d(in_channel, channel_1, (3,3), padding=1),
    nn.BatchNorm2d(channel_1),
    nn.ReLU(),
    nn.Conv2d(channel_1, channel_2, (3,3), padding=1),
    nn.BatchNorm2d(channel_2),
    nn.ReLU(),
    nn.MaxPool2d((2,2)),
    nn.Conv2d(channel_2, channel_3, (3,3), padding=1),
    nn.BatchNorm2d(channel_3),
    nn.ReLU(),
    nn.Conv2d(channel_3, channel_4, (3,3), padding=1),
    nn.BatchNorm2d(channel_4),
    nn.ReLU(),
    nn.MaxPool2d((2,2)),
    nn.Conv2d(channel_4, channel_5, (3,3), padding=1),
    nn.BatchNorm2d(channel_5),
    nn.ReLU(),
    nn.Conv2d(channel_5, channel_6, (3,3), padding=1),
    nn.BatchNorm2d(channel_6),
    nn.ReLU(),
    nn.MaxPool2d((2,2)),
    nn.Conv2d(channel_6, channel_7, (3,3), padding=1),
    nn.BatchNorm2d(channel_7),
    nn.ReLU(),
    nn.Conv2d(channel_7, channel_8, (3,3), padding=1),
    nn.BatchNorm2d(channel_8),
    nn.ReLU(),
    Flatten(),
    nn.Linear(int(channel_8*imH*imW/(4*4*4)), 48),
    nn.BatchNorm1d(48),
    nn.ReLU(),
    nn.Dropout(p=pDropOut),
    nn.Linear(48,6)
)

model.apply(init_weight)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
epochs = 20

tic = time.time()
losses, iters, trainAccs, devAccs = train(model, loaders, optimizer, epochs, print_every)
trainingTime = time.time() - tic
print("Training time was: ", trainingTime, " seconds")

plt.figure()
plt.plot(iters, losses, 'k')
plt.xlabel("Iterations")
plt.ylabel("Loss")

plt.figure()
plt.plot(np.arange(1,21,1), trainAccs[0::2], 'k', label="Training acc.")
plt.plot(np.arange(1,21,1), devAccs[0::2], 'k--', label="Validation acc.")
plt.xticks(np.arange(0, 22, 2))
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Test model
print("Model applied to training set:")
acc_train = check_accuracy(loaders['train'], model)

print("Model applied to development set:")
acc_dev = check_accuracy(loaders['dev'], model)

performErrorAnalysis = False
if performErrorAnalysis:
    error_analysis(model, loaders, stdIm, meanIm, maxNumFotos=200)

#print("Model's state_dict:")
#for param_tensor in model.state_dict():
#    print(param_tensor, "\t", model.state_dict()[param_tensor].size())


#pathStateDict = os.getcwd() + "/20200223_1149.pt"
#torch.save(model.state_dict(), pathStateDict)



