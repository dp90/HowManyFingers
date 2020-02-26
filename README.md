# How many fingers?

This repository provides the tools to train a model to predict the number of fingers held up by a hand in a given image. Two implementations of this model are compared: a PyTorch version and a version, in which the layers, regularizers, normalization, optimizers and loss function are implemented in Numpy. 

## Introduction
An important factor in the practical applicability of deep neural networks is cost. Gathering and storage of data. Duration of training models. 
Numerous efforts into optimization of networks to increase learning speed: different nets, normalization, parallelization, etc. 

The main question is therefore: How does the PyTorch library compare to my own implementation of various standard architectures, regularizers, normalization techniques and optimizers in terms of speed?

## Methods
### PyTorch and Numpy
Some details about the network implementation. 
The method to test is to run both and see how long it takes. 

### Dataset
Pictures with various backgrounds. Pictures with hands from different people. Data augmentation through flipping: left and right hands. 

### Computer specifications
CPU

## Results

## Analysis

## Conclusion
PyTorch is faster, so I wouldn't recommend or personally use my Numpy implementations.

## Discussion
Add data. Other option is to regularize with weight decay/dropout, but this also decreases the training error. Given the Bayesian error of 0.0%, this solution could work to improve the model's performance in a practical application, but is not the optimal solution. 
Another possibility could be to pre-proces the data to remove the background from the images, so that only the hands themselves remain. 
C++.
