# How many fingers?
This repository provides the tools to train a model to predict the number of fingers held up by a hand in a given image. Two implementations of this model are compared: a PyTorch version and a version, in which the layers, regularizers, normalization, optimizers and loss function are implemented in Numpy. 

## Introduction
An important factor in the practical applicability of deep neural networks is cost, which can roughly be divided in gathering and storage of data, and computational costs to train and apply models. Decreasing hardware costs reduce computational costs, but another major driver of deep learning's success is the more efficient use of available hardware.  
Numerous efforts over the last decade have improved computational efficiency: 
- Different networks architectures, such as the residual blocks in ResNet (He et al., 2015) and the inception blocks in GoogLeNet (Szegedy et al., 2014);  
- Parallelization, such as using GPUs or development of special chips; 
- Different weight initializations, such as Xavier (Glorot & Bengio, 2015) or He-normal (He et al., 2015);
- Normalization techniques, such as batch (Ioffe & Szegedy, 2015), layer (Ba et al., 2016) or group normalization (Wu & He, 2018);  
- Different optimizers, such as Adagrad (Duchi et al., 2011), RMSprop (Hinton, 2013) and Adam (Kingma & Ba, 2015).

Another factor in the speed of algorithms, ceteris paribus, is the efficiency of their code. A number of libraries have become available, such as Keras/TensorFlow, PyTorch, Scikit, etc. Besides an easy-to-use interface, these libraries tend to have been optimized for speed. However, it is unknown to the author, if and how much faster the libaries are than a vectorized Numpy version of the same algorithms, coded by an average Python programmer.  
The main question is therefore: *How does the PyTorch library compare to my own implementation of various standard architectures, regularizers, normalization techniques and optimizers in terms of speed?*

To compare the training speed of the Numpy and PyTorch modules, a prediction challenge was required that could be implemented with relative ease, but that would still offer valuable lessons in data gathering and processing. The challenge was found in predicting the number of fingers held up by a hand in a given image, essentially an image classification problem.  
In the subsequent sections, the applied methods are discussed, as well as the results and analysis of these results. Finally, the conclusion is presented, followed by a short discussion of the project and potential improvements. 

## Methods
A model was trained with PyTorch to test various network architectures, after which a model was trained with the same architecture, using the Numpy module. The relative speed of both implementations is computed as the inversed relative time both models required to train.

### Dataset
The dataset consists of 4049 images of hands with 0-5 fingers extended. Most (~80%) of these images are of the author's hand, and the remainder are collected from 11 volunteers in equal proportion. The distribution of the images among each class can be found in table 1. 

Table 1: Distribution of images among classes
| # Fingers     | Percentage of total |
| col 3 is      | right-aligned       |
| col 2 is      | centered            |
| zebra stripes | are neat            |


8098 Pictures with various backgrounds. Pictures with hands from different people. Data augmentation through flipping: left and right hands. 
Preprocessing: squaring the images, cropping and normalization. 

Divided into a training set (80%), development set (10%) and a test set (10%). 
training error was close to the Bayes error, while the validation error was less than 10%. 

### Network architecture


### Numpy implementation
Some details about the network implementation.  

### Computer specifications
CPU

## Results

## Analysis

## Conclusion
PyTorch is faster, so I wouldn't recommend or personally use my Numpy implementations.

## Discussion
Add data. Other option is to regularize with weight decay/dropout, but this also decreases the training error. Given the Bayesian error of 0.0%, this solution could work to improve the model's performance in a practical application, but is not the optimal solution.  
Another possibility could be to pre-proces the data to remove the background from the images, so that only the hands themselves remain.  
Ensemble.  
C++.

## References
K. He, X. Zhang, S. Ren & J. Sun (2015), Deep Residual Learning for Image Recognition.  
C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke & A. Rabinovich (2014), Going Deeper with Convolutions.  
X. Glorot & Y. Bengio (2010), Understanding the Difficulty of Training Deep Feedforward Neural Networks.  
S. Ioffe & C. Szegedy (2015), Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.  
J. L. Ba & J. R. Kiros, G. E. Hinton (2016), Layer Normalization.  
Y. Wu & K. He (2018), Group Normalization.  
J. Duchi, E. Hazan & Y. Singer (2011), Adaptive Subgradient Methods for Online Learning and Stochastic Optimization.  
G. E. Hinton (2013), Neural Networks for Machine Learning - Lecture 6a - Overview of mini-batch gradient descent.  
D. P. Kingma & J. L. Ba (2015), Adam: a Method for Stochastic Optimization.  
