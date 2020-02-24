# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:28:03 2020

@author: David
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

dtype = torch.float32
device = torch.device('cpu')

# From Stanford CS231n
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc

# From Stanford CS231n
def train(model, loaders, optimizer, epochs, print_every):
    model = model.to(device=device)
    losses, iters, acc_train, acc_dev = [], [], [], []
    counter = 0
    for e in range(epochs):
        print('Epoch %d' % (e+1))
        for t, (x, y) in enumerate(loaders['train']):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)

            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t+1, loss.item()))
                acc_train.append( check_accuracy(loaders['train'], model) )
                acc_dev.append( check_accuracy(loaders['dev'], model) )
                print()
            
            counter += 1
            iters.append(counter)
            losses.append(loss.detach()) # loss has autograd history, so detach to avoid memory leak https://pytorch.org/docs/stable/notes/faq.html#my-model-reports-cuda-runtime-error-2-out-of-memory
    return np.asarray(losses), np.asarray(iters), np.asarray(acc_train), np.asarray(acc_dev)

# From Stanford CS231n
def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

# From Stanford CS231n
class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)

def init_weight(m):
    if (type(m) == nn.Linear or type(m) == nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)
    return
    
def error_analysis(model, loaders, stdIm, meanIm, maxNumFotos=200):
    dtype = torch.float32
    device = torch.device('cpu')
    model.eval()
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for x, y in loaders['dev']:
            x = x.to(device=device, dtype=dtype)  # move to device: GPU/CPU
            y = y.to(device=device, dtype=torch.long)
            
            scores = model(x)
            _, preds = scores.max(1)
            
            counter = 0
            for i in range(x.size()[0]):
                if (y[i] != preds[i] and counter < maxNumFotos):
                    counter += 1
                    t = x[i,:,:,:]
                    im = (np.transpose(t.numpy() * stdIm + meanIm,(1,2,0))) / 255
                    plt.figure()
                    plt.imshow(im)
                    plt.title("pred = "+str(preds[i].numpy())+", label = "+str(y[i].numpy()))
            
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        
    return