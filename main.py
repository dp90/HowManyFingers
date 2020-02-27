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
from numpyTools import *
from numpyLayers import *

# Set directories for folder with new and prepared images
imPathOrig = os.getcwd() + "/newImages/"
imPathPrep = os.getcwd() + "/imagesPrepared/"

imH, imW = 64, 64 # processed image dimensions
areNewImagesAvailable = False
if areNewImagesAvailable:
    add_images_to_dataset(imPathOrig,imPathPrep,imSize=(imH,imW),save=True,flips=True)

data, meanIm, stdIm = load_data_sets(imPathPrep, split=(80,10,10), imSize=(imH,imW)) # dictionary

# Print statistics on number of samples per class
yTrain, yDev, yTest = data['Y_train'], data['Y_dev'], data['Y_test']
y = np.concatenate((yTrain, yDev, yTest), axis=0)
for i in range(6):
    print(np.sum(y == i) / y.shape[0])
    print(np.sum(y == i))

# Select the use of PyTorch "= True", or Numpy "= False"
usePyTorch = True
if usePyTorch:
    batch_size = 256
    loaders = numpy_to_pytorch(data, batch_size)
    del data
    model = build_pytorch_model()
    
    print_every = 26
    learning_rate = 3e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 25

    tic = time.time()
    losses, iters, trainAccs, devAccs = train(model, loaders, optimizer, epochs, print_every)
    trainingTime = time.time() - tic
    print("Training time was: ", trainingTime, " seconds")
    
    performErrorAnalysis = False
    results_pytorch(iters,losses,trainAccs,devAccs,loaders,model,stdIm,meanIm,performErrorAnalysis)

    #pathStateDict = os.getcwd() + "/20200223_1149.pt"
    #torch.save(model.state_dict(), pathStateDict)
    
elif usePyTorch is False:
    num_train = 128
    small_data = {
      'X_train': data['X_train'],
      'Y_train': data['Y_train'],
      'X_val': data['X_dev'],
      'Y_val': data['Y_dev'],
      'X_test': data['X_test'],
      'Y_test': data['Y_test'],
    }
    del data
    architecture = build_architecture()
    
    learning_rate = 3e-3
    model = NeuralNet(architecture, input_dim = 3*64*64, num_classes = 6,
                  dtype=np.float64)
    
    solver = Solver(model, small_data,
                    print_every=1, num_epochs=25, batch_size=np.min((num_train,128)),
                    update_rule='adam',
                    optim_config={
                      'learning_rate': learning_rate,
                    }
             )
    tic = time.time()
    solver.train()
    trainingTime = time.time() - tic
    
    trainAccHist = solver.train_acc_history
    devAccHist = solver.val_acc_history
    lossHist = solver.loss_history
    
    plt.figure()
    plt.plot(lossHist, 'k')
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    
    plt.figure()
    plt.plot(trainAccHist, 'k', label="Training acc.")
    plt.plot(devAccHist, 'k--', label="Validation acc.")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()


# Statistics on improvement development accuracy for increasing dataset size
'''
nSamples = np.asarray([16, 32, 64, 128, 256, 512, 1024, 2048, 8098])
accTrainSamples = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 0.998, 0.998, 0.985, 0.981])
accDevSamples = np.asarray([0.188, 0.281, 0.328, 0.359, 0.426, 0.523, 0.672, 0.743, 0.898])

plt.figure()
plt.plot(nSamples, 1-accTrainSamples, 'k', label="Training data")
plt.plot(nSamples, 1-accDevSamples, 'k--', label="Validation data")
plt.xlabel("Samples in dataset")
plt.ylabel("Error")
plt.ylim(0,0.9)
plt.legend()
'''