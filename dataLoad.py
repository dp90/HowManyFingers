# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 10:03:48 2020

@author: David
"""

import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def labels_into_dict(srcDir):
    d = {}
    with open(srcDir + "labels.txt", "r") as labelFile:
        for line in labelFile:
            key, value = line.strip().split(':-')
            d[key] = value
        labelFile.close()
    return d

def imgs_to_array(srcDir,imSize):
    """
    Parameters
    ----------
    srcDir : STRING
        Directory to image files to be included in dataset X. 

    Returns
    -------
    X : NUMPY ARRAY
        Array with RGB values of each pixel in N images.
        Shape: N x C x H x W.
    """
    listFiles = [fileName for fileName in os.listdir(srcDir) if (fileName != "Thumbs.db") and (fileName != "labels.txt")]
    labelDict = labels_into_dict(srcDir)
    X = np.zeros((len(listFiles),3,imSize[0],imSize[1]))
    Y = np.zeros((len(listFiles)))
    for i, fileName in enumerate(listFiles):
        img = Image.open(srcDir + fileName)
        imArray = np.asarray(img)
        imArray = np.transpose(imArray, (2,0,1)) # shape C x H x W
        X[i,:,:,:] = imArray
        Y[i] = int(labelDict[fileName])
    
    return X, Y

def shuffle_data(X, Y):
    N = X.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    
    return X[idx,:,:,:], Y[idx]

def split_data_sets(X, Y, split):
    """
    Parameters
    ----------
    X : NUMPY ARRAY
        RGB pixel data for all images in dataset.
        Shape: N x C x H x W
    split : TUPLE
        Contains respectively the sizes of the train, development and 
        test sets as a percentage of the entire dataset.
        Example: (80,10,10) means training set contains 80% of total 
        number of samples, etc.

    Returns
    -------
    X_train : NUMPY ARRAY
        RGB pixel data for all images in training set.
        Shape: N_train x C x H x W
    X_dev : NUMPY ARRAY
        RGB pixel data for all images in development set.
        Shape: N_dev x C x H x W.
    X_test : NUMPY ARRAY
        RGB pixel data for all images in test set.
        Shape: N_test x C x H x W.
    """
    N = X.shape[0]
    split = (80,10,10)
    assert np.sum(split)==100, "Make sure your split percentages add up to 100 to use all available data."
    pctTrain, pctDev, pctTest = split
    nTrain, nDev = int(pctTrain/100*N), int(pctDev/100*N)
    X_train = X[0:nTrain,:,:,:]
    Y_train = Y[0:nTrain]
    X_dev = X[nTrain : nTrain + nDev,:,:,:]
    Y_dev = Y[nTrain : nTrain + nDev]
    X_test = X[nTrain + nDev : ,:,:,:]
    Y_test = Y[nTrain + nDev :]
    
    return X_train, X_dev, X_test, Y_train, Y_dev, Y_test

def normalize_data(X_train, X_dev, X_test):
    mean_image = np.mean(X_train, axis=0)
    std_image = np.std(X_train, axis=0)
    X_train = (X_train - mean_image) / std_image
    X_dev = (X_dev - mean_image) / std_image
    X_test = (X_test - mean_image) / std_image
    
    return X_train, X_dev, X_test, mean_image, std_image

def load_data_sets(srcDir, split, imSize):
    X,Y = imgs_to_array(srcDir,imSize)
    X,Y = shuffle_data(X,Y)
    X_train, X_dev, X_test, Y_train, Y_dev, Y_test = split_data_sets(X, Y, split)
    X_train, X_dev, X_test, meanIm, stdIm = normalize_data(X_train, X_dev, X_test)
    data = {'X_train': X_train, 'X_dev': X_dev, 'X_test': X_test, 
            'Y_train': Y_train, 'Y_dev': Y_dev, 'Y_test': Y_test}

    return data, meanIm, stdIm

class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)

def numpy_to_pytorch(data, batch_size):
    datasetTrain = MyDataset(data['X_train'], data['Y_train'])
    datasetDev = MyDataset(data['X_dev'], data['Y_dev'])
    datasetTest = MyDataset(data['X_test'], data['Y_test'])
    loaderTrain = DataLoader(datasetTrain, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=torch.cuda.is_available() )
    loaderDev = DataLoader(datasetDev, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=torch.cuda.is_available() )
    loaderTest = DataLoader(datasetTest, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=torch.cuda.is_available() )
    loaders = {'train': loaderTrain, 'dev': loaderDev, 'test': loaderTest}
    return loaders


