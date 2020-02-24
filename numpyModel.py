#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 16:06:17 2020

@author: david
"""

import numpy as np

from builtins import range
from builtins import object

from cs231n.layers import *
from cs231n.layer_utils import *


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
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
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        for i in range(self.num_layers):
            if i == 0:
                self.params['W'+str(i+1)]     = np.random.normal(0, weight_scale, (input_dim,hidden_dims[i]))
                self.params['b'+str(i+1)]     = np.zeros(hidden_dims[i])
                self.params['gamma'+str(i+1)] = np.ones(hidden_dims[i])
                self.params['beta'+str(i+1)]  = np.zeros(hidden_dims[i])
            elif i == self.num_layers-1:
                self.params['W'+str(i+1)]     = np.random.normal(0, weight_scale, (hidden_dims[i-1],num_classes))
                self.params['b'+str(i+1)]     = np.zeros(num_classes)
            else:
                self.params['W'+str(i+1)]     = np.random.normal(0, weight_scale, (hidden_dims[i-1],hidden_dims[i]))
                self.params['b'+str(i+1)]     = np.zeros(hidden_dims[i])
                self.params['gamma'+str(i+1)] = np.ones(hidden_dims[i])
                self.params['beta'+str(i+1)]  = np.zeros(hidden_dims[i])
        
        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.ln_params = [{} for i in range(self.num_layers - 1)]
            
        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        if self.normalization=='layernorm':
            for ln_param in self.ln_params:
                ln_param['eps'] = 1e-5
                        
        input_next_layer = X
        caches = {}
        
        for i in range(self.num_layers-1):
            if self.normalization=='batchnorm':
                [input_next_layer, current_cache] = affine_bn_relu_forward(input_next_layer, self.params['W'+str(i+1)], 
                                                                          self.params['b'+str(i+1)], 
                                                                          self.params['gamma'+str(i+1)], 
                                                                          self.params['beta'+str(i+1)], self.bn_params[i])
            elif self.normalization==None:
                [input_next_layer, current_cache] = affine_relu_forward(input_next_layer, self.params['W'+str(i+1)], 
                                                                        self.params['b'+str(i+1)])

            if self.use_dropout:
                [input_next_layer, dropout_cache] = dropout_forward(input_next_layer, self.dropout_param)
                caches['do_cache'+str(i+1)] = dropout_cache
            caches['cache'+str(i+1)] = current_cache
        
        [scores, scores_cache] = affine_forward(input_next_layer, self.params['W'+str(self.num_layers)], 
                                                self.params['b'+str(self.num_layers)])

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}

        [loss, grad_softmax] = softmax_loss(scores,y)
        for i in range(self.num_layers):
            loss += self.reg*0.5*np.sum(self.params['W'+str(i+1)]**2)
        
        [dL, dW, db] = affine_backward(grad_softmax, scores_cache)
        dW += self.reg*self.params['W'+str(self.num_layers)]
        grads['W'+str(self.num_layers)] = dW
        grads['b'+str(self.num_layers)] = db
        
        for i in reversed(range(self.num_layers-1)):
            if self.use_dropout:
                dL = dropout_backward(dL, caches['do_cache'+str(i+1)])
            if self.normalization=='batchnorm':
                [dL, dW, db, dgamma, dbeta] = affine_bn_relu_backward(dL, caches['cache'+str(i+1)])
                dW += self.reg*self.params['W'+str(i+1)]
                grads['W'+str(i+1)] = dW
                grads['b'+str(i+1)] = db
                grads['beta'+str(i+1)] = dbeta
                grads['gamma'+str(i+1)] = dgamma
            elif self.normalization==None:
                [dL, dW, db] = affine_relu_backward(dL, caches['cache'+str(i+1)])
                dW += self.reg*self.params['W'+str(i+1)]
#                 db += self.reg*self.params['b'+str(i+1)]
                grads['W'+str(i+1)] = dW
                grads['b'+str(i+1)] = db
                grads['beta'+str(i+1)] = 0
                grads['gamma'+str(i+1)] = 0

        return loss, grads

    
    
    
    
    
    
    
    
    