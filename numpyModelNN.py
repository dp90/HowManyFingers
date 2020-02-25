#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 12:12:32 2020

@author: david
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 16:06:17 2020

@author: david
"""

import numpy as np

from builtins import range
from builtins import object

from numpyLayers import *

class NeuralNet(object):
    def __init__(self, architecture, input_dim=3*64*64, num_classes=6,
                 dtype=np.float32):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.architecture = architecture # list with layers
        self.nLayers = len(architecture)
        self.dtype = dtype
        self.params = {}
        self.bn_params = {}

        layersWithoutWeights = ['ReLU','Dropout','Maxpool2D']
        for i, layer in enumerate(self.architecture):
            if layer[0] in layersWithoutWeights:
                continue
            elif layer[0] == 'Conv2D':
                filterSizes = layer[1][0:4]
                kaiming = np.sqrt(2. / np.prod(filterSizes[1:]))
                self.params['W'+str(i)] = np.random.normal(0, kaiming, filterSizes)
                self.params['b'+str(i)] = np.zeros(filterSizes[0])
            elif layer[0] == 'Affine':
                kaiming = np.sqrt(2. / layer[1][0])
                self.params['W'+str(i)] = np.random.normal(0, kaiming, layer[1])
                self.params['b'+str(i)] = np.zeros(layer[1][1])
            elif (layer[0] == 'Batchnorm1D') or (layer[0] == 'Batchnorm2D'):
                self.params['gamma'+str(i)] = np.ones(layer[1])
                self.params['beta'+str(i)] = np.zeros(layer[1])
            else:
                print(layer[0], " not recognized as valid layer type.")
        
        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
        
        self.bn_params = [{'mode': 'train'} for i in range(self.nLayers - 1)]
        
    def loss(self, X, Y=None):
        """
        Compute loss and gradient for the fully-connected net.
        """
        X = X.astype(self.dtype)
        mode = 'test' if Y is None else 'train'

        for bn_param in self.bn_params:
            bn_param['mode'] = mode

        inNextL = X
        caches = {}
        
        for i, layer in enumerate(self.architecture[:-1]):
            if layer[0] == "Conv2D":
                [inNextL, cache] = conv2D_forward(inNextL, self.params['W'+str(i)], self.params['b'+str(i)], layer[1][-1])
            elif layer[0] == "Affine":
                [inNextL, cache] = affine_forward(inNextL, self.params['W'+str(i)], self.params['b'+str(i)])
            elif layer[0] == "Batchnorm1D":
                [inNextL, cache] = batchnorm1D_forward(inNextL, self.params['gamma'+str(i)], self.params['beta'+str(i)], self.bn_params[i])
            elif layer[0] == "Batchnorm2D":
                [inNextL, cache] = batchnorm2D_forward(inNextL, self.params['gamma'+str(i)], self.params['beta'+str(i)], self.bn_params[i])
            elif layer[0] == "ReLU":
                [inNextL, cache] = relu_forward(inNextL)
            elif layer[0] == "Dropout":
                dropout_params = {'mode': mode, 'p': layer[1]}
                [inNextL, cache] = dropout_forward(inNextL, dropout_params)
            elif layer[0] == "Maxpool2D":
                pool_param = {'pool_height': layer[1][0], 'pool_width': layer[1][1], 'stride': layer[1][2]}
                [inNextL, cache] = maxpool2D_forward(inNextL, pool_param)
            caches[str(i)] = cache
        
        [scores, scores_cache] = affine_forward(inNextL, self.params['W'+str(self.nLayers-1)], 
                                                self.params['b'+str(self.nLayers-1)])

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}

        [loss, grad_softmax] = softmax_loss(scores,Y)
        #for i in range(self.num_layers):
        #    loss += self.reg*0.5*np.sum(self.params['W'+str(i+1)]**2)
        
        [dL, dW, db] = affine_backward(grad_softmax, scores_cache)
        #dW += self.reg*self.params['W'+str(self.num_layers)]
        grads['W'+str(self.nLayers-1)] = dW
        grads['b'+str(self.nLayers-1)] = db
        
        for i in reversed(range(self.nLayers-1)):
            cache = caches[str(i)]
            layer = self.architecture[i]
            if layer[0] == "Conv2D":
                dL, grads['W'+str(i)], grads['b'+str(i)] =  conv2D_backward(dL, cache)
            elif layer[0] == "Affine":
                dL, grads['W'+str(i)], grads['b'+str(i)] = affine_backward(dL, cache)
            elif layer[0] == "Batchnorm1D":
                dL, grads['gamma'+str(i)], grads['beta'+str(i)] = batchnorm1D_backward(dL, cache)
            elif layer[0] == "Batchnorm2D":
                dL, grads['gamma'+str(i)], grads['beta'+str(i)] = batchnorm2D_backward(dL, cache)
            elif layer[0] == "ReLU":
                dL = relu_backward(dL, cache)
            elif layer[0] == "Dropout":
                dL = dropout_backward(dL, cache)
            elif layer[0] == "Maxpool2D":
                dL = maxpool2D_backward(dL, cache)

        return loss, grads

    
    
    
    
    
    
    
    
    