#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:42:06 2020

@author: david
"""

import numpy as np

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """

    xvec = np.reshape(x, (x.shape[0], -1)) # size: N x d1*d2*...*dk
    out = xvec@w + b[None,:]
    cache = (x, w, b)
    
    return out, cache

def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    
    dx = dout@(w.T)
    dx = np.reshape(dx,x.shape)
    dw = (np.reshape(x, (x.shape[0], -1)).T).dot(dout)
    db = np.sum(dout,0)
    
    return dx, dw, db

def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    
    cache = x
    out = x
    out[out<0] = 0

    return out, cache

def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """

    dx = (cache>0)*dout

    return dx

def batchnorm1D_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift parameter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    if mode == 'train':

        sample_mean = np.mean(x, 0)
        sample_var = np.var(x, 0)
        x_norm = (x - sample_mean[None,:]) / (np.sqrt(sample_var[None,:] + eps))
        out = gamma*x_norm + beta
        
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        
        cache = {'x': x, 'x_norm': x_norm, 'var': sample_var,
                 'gamma': gamma, 'eps': eps}
        
    elif mode == 'test':

        x_norm = (x - running_mean[None,:]) / (np.sqrt(running_var[None,:] + eps))
        out = gamma*x_norm + beta

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm1D_backward(dout, cache):
    """
    Backward pass for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """    
    x, x_norm = cache['x'], cache['x_norm']
    N = x.shape[0]
    var = cache['var']
    gamma, eps = cache['gamma'], cache['eps']
    
    dbeta = np.sum(dout, 0)              # size: 1 x D
    dgamma = np.sum(x_norm*dout, 0)      # size: 1 x D
    dx_norm = gamma*dout                 # size: N x D    
    dvar =  -np.sum(dx_norm*x_norm,0)    
    dx = (var + eps)**(-0.5)*( N*dx_norm + dvar*x_norm + -np.sum(dx_norm,0) ) / N # size: N x D

    return dx, dgamma, dbeta



def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter - neuron output kept with probability p
      - mode: 'test' or 'train' - if 'test', nothing is done
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']

    if mode == 'train':
        mask = (np.random.rand(*x.shape) < p) / p
        out = x*mask
    elif mode == 'test':
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache

def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']
    if mode == 'train':
        dx = mask*dout    
    elif mode == 'test':
        dx = dout
    
    return dx

def conv2D_forward(x, w, b, conv_param):
    """
    Perform the forward pass for a convolutional layer.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
        
    nPad = conv_param['pad']
    stride = conv_param['stride']
    F, C, HH, WW = w.shape
    N, _, H, W = x.shape
    Hout = int(1 + (H + 2 * nPad - HH) / stride)
    Wout = int(1 + (W + 2 * nPad - WW) / stride)
    
    x_cols = im2col_indices(x,HH,WW,padding=nPad,stride=stride)
    w_cols = w.reshape(F,-1)
    
    out1 = (w_cols.dot(x_cols) + b[:,None]).reshape(1,-1)
    out2 = out1.reshape(N,-1,order='F')
    out = out2.reshape(N,F,Hout,Wout)
    cache = (x, w, b, conv_param)
    
    return out, cache

def conv2D_backward(dout, cache):
    """
    Backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """    
    
    x, w, b, conv_param = cache
    nPad = conv_param['pad']
    stride = conv_param['stride']
    F, C, HH, WW = w.shape
    N, _, H, W = x.shape
    
    x_cols = im2col_indices(x,HH,WW,padding=nPad,stride=stride)
    w_cols = w.reshape(F,-1)
    
    dout = dout.reshape(N,F,-1)           # N x F x H'*W'
    dout = dout.transpose((1,0,2))        # F x N x H'*W'
    dout = dout.reshape(F,-1,order='F')   # F x N*H'*W' = w_col.dot(x_col).shape
    
    dw = dout.dot(x_cols.T)
    db = np.sum(dout,1)
    dx = (w_cols.T).dot(dout)
    
    dx = col2im_indices(dx, x.shape, field_height=HH, field_width=WW, padding=nPad,
                           stride=stride)
    dw = dw.reshape(F, C, HH, WW)

    return dx, dw, db


def maxpool2D_forward(x, pool_param):
    """
    Forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """

    stride = pool_param['stride']
    HH, WW = pool_param['pool_height'], pool_param['pool_width']
    N, C, H, W = x.shape
    Hout = int(1 + (H - HH) / stride)
    Wout = int(1 + (W - WW) / stride)

    x_cols = im2col_indices(x,HH,WW,padding=0,stride=stride)
    x_cols = (x_cols.T).reshape(int(N*C*H*W/(HH*WW)),int(HH*WW))
    out1 = np.max(x_cols,1).reshape(HH*WW,-1).T
    out2 = out1.reshape(N, C, -1)
    out = out2.reshape(N,C,Hout,Wout)
    cache = (x, pool_param)
    
    return out, cache

def maxpool2D_backward(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """

    x, pool_param = cache
    
    stride = pool_param['stride']
    HH, WW = pool_param['pool_height'], pool_param['pool_width']
    N, C, H, W = x.shape
    
    x_cols = im2col_indices(x,HH,WW,padding=0,stride=stride).T
    x_cols = x_cols.reshape(int(N*C*H*W/(HH*WW)),int(HH*WW))
        
    dout_cols = dout.transpose(3,0,1,2).ravel()
    mask = np.argmax(x_cols,1)
    max_pos = np.zeros(x_cols.shape)
    max_pos[range(x_cols.shape[0]),mask] = np.squeeze(dout_cols)
    max_pos = max_pos.reshape(int(N*H*W/(HH*WW)),int(C*HH*WW))
    max_pos = max_pos.T
    
    dx1 = col2im_indices(max_pos, x.shape, field_height=HH, field_width=WW, padding=0,
                         stride=stride)
    dx = dx1.reshape(x.shape)

    return dx


def batchnorm2D_forward(x, gamma, beta, bn_param):
    """
    Forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    
    N, C, H, W = x.shape
    x = np.transpose(x, (1,0,2,3))
    x_flat = x.reshape(C,-1)
    x_norm_flat, cache = batchnorm_forward(x_flat.T, gamma, beta, bn_param)
    x_norm = (x_norm_flat.T).reshape(C,N,H,W)
    out = np.transpose(x_norm, (1,0,2,3))

    return out, cache

def batchnorm2D_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    
    N, C, H, W = dout.shape
    dout = np.transpose(dout, (1,0,2,3))
    dout_flat = dout.reshape(C,-1)
    dx_flat, dgamma, dbeta = batchnorm_backward(dout_flat.T, cache)
    dx = (dx_flat.T).reshape(C,N,H,W)
    dx = np.transpose(dx, (1,0,2,3))
    
    return dx, dgamma, dbeta

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    
    Provided in Standford CS231n
    """
    
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    
    return loss, dx

# Functions get_im2col_indices(), im2col_indices() and col2im_indices() from
# Stanford course CS231n
from builtins import range

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int( (H + 2 * padding - field_height) / stride + 1 )
    out_width = int( (W + 2 * padding - field_width) / stride + 1 )

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
    
    return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride)
    
    cols = x_padded[:, k, i.astype(int), j.astype(int)]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                                 stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]
























