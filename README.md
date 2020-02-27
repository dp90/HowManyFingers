# How many fingers?
This repository provides the tools to train a model to predict the number of fingers held up by a hand in a given image. Two implementations of this model are compared: a PyTorch version and a version, in which the layers, regularizers, normalization, optimizers and loss function are implemented in Numpy. 
Explanation of files in folder

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

To compare the training speed of the Numpy and PyTorch modules, a prediction challenge was required that could be implemented with relative ease, but that would still offer valuable lessons in data gathering and processing. The challenge was found in predicting the number of fingers held up by a hand in a given image, essentially an image classification problem. The relative speed of both implementations is computed as the inverse of the relative time both models require to train.

In the subsequent sections, the applied methods are discussed, as well as the results and analysis of these results. Finally, the conclusion is presented, followed by a short discussion of the project and potential improvements. 

## Methods
This section describes details of the dataset used to train and test the model, it discusses the applied network architecture and the method applied to find it, and concludes with remarks on the Numpy module and the machine the model is run on. The relative speed of both implementations is computed as the inverse of the relative time both models require to train. 

### Dataset
The dataset consists of 4049 images of both palms and backs of hands with 0-5 fingers extended (see figure XXX). Most (~80%) of these images are of the author's hand, and the remainder is collected from 11 volunteers in equal proportion. 

Figure 1: Examples of images in dataset  
![alt text](https://github.com/dp90/HowManyFingers/blob/master/Images/Example1.png "Example image")
![alt text](https://github.com/dp90/HowManyFingers/blob/master/Images/Example2.png "Example image")  
![alt text](https://github.com/dp90/HowManyFingers/blob/master/Images/Example3.png "Example image")

The distribution of the images amongst the classes can be found in table 1. The images were captured with various backgrounds and in diverse lighting conditions to obtain a degree of generalizability. As most images were taken with the right hand, the images were flipped to represent both left and right hands, and additionally to double the dataset size. 

Table 1: Distribution of images amongst classes
| # Fingers     | Percentage of total |
| ------------- |:-------------------:|
|       0       |       7.9%          |
|       1       |       17.7%         |
|       2       |       24.2%         |
|       3       |       14.0%         |
|       4       |       23.4%         |
|       5       |       12.8%         |

After collection and labelling of the data, the images were squared by cropping out 1/8 or 7/32 of the pixels on both sides of the longest edge for aspect ratios of 3:4 and 9:16 respectively. To obtain workable samples, the images were compressed to 64x64 pixels with Python's PIL module. A test with an early stage, possibly suboptimal model reveiled that compressing the images to 32x32 pixels significantly reduced the model's generalizability.

To train, develop and test the model, the dataset was divided into a training, validation and test set. These contain respectively 80%, 10% and 10% of the images, or 6478, 809 and 811 images in absolute terms.  

### Network architecture
A suitable architecture was found with PyTorch and subsequently used to test the relative speed of PyTorch and the Numpy module. But what is a suitable architecture? The Bayes error for this problem is 0%, so that the training error should be approximately that. The error for the validation set is found to be sufficiently low at less than 10%. A matching architecture was found by trail and error, based on informed guesses. A description of this process can be found below, as well as the final architecture. 

After collecting 640 images, an initial architecture was applied of
4 convolutional layers with:
- Batch normalization
- ReLU activation
- Filter size: 3x3
- Stride: 1
- Padding: 1
- Channels: 16, 16, 32 and 32 respectively

Followed by
Affine layer with:
- 40 hidden units
- Batch normalization
- ReLU activation
- Drop out: p = 0.5

Followed by
Affine layer and softmax classifier.

The resulting training and validation set accuracies were approximately 100% and 50%. From these statistics it was concluded that the model suffered from high variance, which can be mitigated by regularization or adding data. As the former at this point seemed unlikely to result in a validation accuracy of over 90%, it was opted to add data.  
Adding roughly 1400 images to end up with 2000 images, increased the validation accuracy to approximately 74%. Encouraged by this improvement, another 2000 images were gathered to achieve a dissapointing 78% validation accuracy. Considering the marginal increase in accuracy with a doubling of the dataset size, the model clearly suffered from high bias, which called for a different architecture.  
I increased the number of layers to allow for further developed feature combinations (entire hands instead of fingers). I increased the number of filters to allow for more feature combinations, e.g. different background colors. I increased the number of hidden units in the affine layer. I added some regularization in the form of drop out. Finally, I added maxpool layers, which made the largest difference. As I apply 3x3 filters to a 128x128 image, the 'filtered' area, even over multiple convolutional layers is relatively small compared to the image dimensions. As the maxpool layer essentially extracts whether certain filters were activated over a given area (2x2 in this case), the information of a layer is kept, while its dimensions are reduced. The subsequent 3x3 filters then effectively cover a much larger area of the original image than they do without the maxpool layer.  
The final architecture is then
- Convolutional layer - channels: 16, filter size: 3x3, stride: 1, padding: 0
- Batch normalization
- ReLU
- Convolutional layer - channels: 16, filter size: 3x3, stride: 1, padding: 0
- Batch normalization
- ReLU
- Maxpool - size: 2x2, stride: 2
- Convolutional layer - channels: 32, filter size: 3x3, stride: 1, padding: 0
- Batch normalization
- ReLU
- Convolutional layer - channels: 32, filter size: 3x3, stride: 1, padding: 0
- Batch normalization
- ReLU
- Maxpool - size: 2x2, stride: 2
- Convolutional layer - channels: 64, filter size: 3x3, stride: 1, padding: 0
- Batch normalization
- ReLU
- Convolutional layer - channels: 64, filter size: 3x3, stride: 1, padding: 0
- Batch normalization
- ReLU
- Maxpool - size: 2x2, stride: 2
- Convolutional layer - channels: 128, filter size: 3x3, stride: 1, padding: 0
- Batch normalization
- ReLU
- Convolutional layer - channels: 128, filter size: 3x3, stride: 1, padding: 0
- Batch normalization
- ReLU
- Affine - hidden units: 48
- Batch normalization
- ReLU
- Drop out - p = 0.75 (probability of being dropped)
- Affine - hidden units: 6
- Softmax

### Numpy implementation
The Numpy implementation of the elements in the final architecture were all part of Stanford CS231n course. The Solver class was provided as course material, but the remainder is my own work, inlcuding the vectorized implementations of the various layers. In the files provided in this repository it is specified what is and what is not my own work. 

### Computer specifications
As the Numpy implementations are by no means optimized for GPU use, both model implementations are run on CPU of type Intel Core i7-4700MQ. The operating system is Linux Mint v19.3. 

## Results & Analysis
This section treats the training and test results of both model implementations, as well as their relative speed. Additionally, some examples are presented of misclassifications to get a more tangible understanding of the model's performance. 

### PyTorch & Numpy
As shown in figure 2, the accuracies of the training and validation sets increase in tandem up to approximately 70%. From there onwards, the difference increases to a final gap of approximately 10%. 

Figure 2: PyTorch model - training and validation accuracies (left) and losses (right) per iteration
![alt text](https://github.com/dp90/HowManyFingers/blob/master/Images/AccPlotPyTorch.png "Train and validation accuracies")
![alt text](https://github.com/dp90/HowManyFingers/blob/master/Images/LossPlotPyTorch.png "Losses")  

The Numpy implementation of the model shows similar results (see figure 3). 

Figure 3: Numpy model - train and validation accuracies (left) and losses (right) per iteration
![alt text](https://github.com/dp90/HowManyFingers/blob/master/Images/AccPlotNumpy.png "Train and validation accuracies")
![alt text](https://github.com/dp90/HowManyFingers/blob/master/Images/LossPlotNumpy.png "Losses")  

The final accuracies are similar as expected (table 2). Due to the randomness in the weight initialization, the models are not identical. 

Table 2: Accuracies of PyTorch and Numpy model implementations
|    Module     | Training accuracy | Validation accuracy | Test accuracy |
| ------------- |:-----------------:| :------------------:|:-------------:|
|     PyTorch   |       99.01%      |        90.98%       |     90.51%    |
|      Numpy    |       98.10%      |        90.73%       |     89.77%    |

### PyTorch Vs Numpy
While the PyTorch and Numpy implementations of the model perform similarly, as expected, in terms of accuracy, their training times are far from similar. While the PyTorch implementations trained in **1146s**, the Numpy implementation took **28291s** (~8h), which makes the PyTorch module **24.7** times as fast! 

### Misclassifications
Before moving on to the conclusions, viewing some of the misclassified images gives more insight in the model's behavior, its abilities, but mostly its sometimes intriguing inabilities. Figure 4 shows three images that are understandably misclassified. The most left image has label 4, but the middle and ring finger are hardly visbile, so that the prediction 2 seems plausible. In the middle picture, it is not difficult to imagine that the classifier counted the thumb too. The color of the fingers in the most right image are very similar to the background, so that the misclassification is not entirely surprising. 

Figure 4: Understandbly misclassified images  
![alt text](https://github.com/dp90/HowManyFingers/blob/master/Images/ErrorPyTorch10.png "Misclassified image")
![alt text](https://github.com/dp90/HowManyFingers/blob/master/Images/ErrorPyTorch2.png "Misclassified image")
![alt text](https://github.com/dp90/HowManyFingers/blob/master/Images/ErrorPyTorch8.png "Misclassified image")

On the other hand, the model misclassified a (larger) number of images that the average human being would have classified correctly without a doubt. Three examples of such images are displayed in Figure 5. Only the little finger in the most right image is not clearly visible, but the predicted value 1 is still off mark, if that finger was not counted. While other network architectures sometimes yielded a high error rate for images with, for example, dark backgrounds, the reported model does not incorporate such biases. 

Figure 5: Strangely misclassified images  
![alt text](https://github.com/dp90/HowManyFingers/blob/master/Images/ErrorPyTorch3.png "Misclassified image")
![alt text](https://github.com/dp90/HowManyFingers/blob/master/Images/ErrorPyTorch5.png "Misclassified image")
![alt text](https://github.com/dp90/HowManyFingers/blob/master/Images/ErrorPyTorch7.png "Misclassified image")

While the results in table 2 show that the validation and test accuracies are decent, the model's performance is not yet at human level, as observed in figure 5. To improve the model, various strategies can be applied. To select a strategy based on information, instead of on a guess, figure 6 shows the development of the training and validation accuracies for an increasing dataset size. It can be concluded that the model suffers from high variance, for which solutions are discussed in the next section. 

Figure 5: Training and validation accuracies for increasing dataset size
![alt text](https://github.com/dp90/HowManyFingers/blob/master/Images/ErrorDataSetSize.png "Training and validation accuracies")

## Conclusion & Discussion
To address the main question of this research project, the PyTorch module outperforms the Numpy implementation by a factor of 24.7. I would therefore not recommend or personally use my Numpy implementations for deep neural networks. While the performance of the Numpy module could still be improved with Cython implementations, JIT and other optimization tools, PyTorch's back-end is written in C++, so that it is unlikely to ever outperform the latter. To improve training time in general, it is advised to make use of GPUs.  
The prediction model itself was shown to achieve a test error of approximately 10%, which is not bad, but still far from the Bayes error of 0.0%. Methods to improve its performance are multiple. A final analysis showed that the model suffers from high variance, which can be resolved by collecting more data or increasing regularization with weight decay or drop out. Even though this solution could work to improve the model's performance in a practical application, it is not the optimal solution. Regularization tends to increase the training error, while that should approximate the Bayes error in an optimal scenario.  
Another possibility to improve the model, perhaps even without adding data, could be to pre-proces the data to remove the background from the images, so that only the hands themselves remain. The model then does not need to learn about the different background that the images can have. Finally, the predictive capacity can be improved by using an ensemble of models.  

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
