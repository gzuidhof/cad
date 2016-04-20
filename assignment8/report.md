

### Assignment week 8, Using Convolutional Neural Networks for classification of CIFAR-10 dataset

Guido Zuidhof (s4160703), Steven Reitsma (s4132343), Robbert van der Gugten (s4137140).  
15/4/2016

----
## Implementation
We refer to our [Git repository](https://github.com/gzuidhof/cad) for our source code. We make use of the following dependencies:
* **Python 2.7**
* **OpenCV2**: augmentation of images
* **cPickle**: read and write files
* **tqdm**: progress bars
* **theano & lasagne**: framework for CNNs

## Initial network
Our network's architecture:
* **Convolution layer** - 32 filters, filter size 5x5, ReLU, weights initialized uniform Glorot
* **Max Pooling layer** - filter size 2x2
* **Convolution layer** - 64 filters, filter size 3x3, ReLU, weights initialized uniform Glorot
* **Max Pooling layer** - filter size 2x2
* **Fully connected layer** - 1024 units, ReLU, weights initialized uniform Glorot
* **Softmax layer** - 10 outputs, one for each class

This architecture resulted in a validation accuracy of 66% (after making the input null mean and unit variance).


## Learning rate optimization & Choosing the update rule
Our initial learning rate is 0.01. We set the batch size to 64 resulting in less noisy gradients. We chose Stochastic Gradient Descent as the initial update rule. We improved this by adding momentum for faster convergence.

 If the learning rate is too high the optimizer will overshoot local optima. If the learning rate is too low, the optimization process will be slow. We tested learning rates of 0.1, 0.01 and 0.001 on an intermediate network to illustrate this.

 After 20 epochs a learning rate of 0.1 resulted in an accuracy of 73.1%, a learning rate of 0.01 resulted in 80.7% and a learning rate of 0.001 achieved 75.2%. We tried the update rule ADAM (Adaptive Moment Estimation), which optimizes the learning rate for each parameter separately. This did not result in a better accuracy (70.2% test accuracy as opposed to the SGD's 76%) for the simple network with data augmentation and batch normalization.

 Learning rate 0.01 (0.807 accuracy)
 ![](https://raw.githubusercontent.com/gzuidhof/cad/master/assignment8/2denseLeaky_undeep_001Learn_807.png)

 Learning rate 0.001 (0.752 accuracy)
 ![](https://raw.githubusercontent.com/gzuidhof/cad/master/assignment8/2dense_Leaky_undeep_0001Learn_752.png)

 Learning rate 0.1 (0.731 accuracy)
 ![](https://raw.githubusercontent.com/gzuidhof/cad/master/assignment8/2dense_leaky_undeep_01Learn_731.png)


## Improving the architecture
We observed that the difference between our validation loss and training loss was small. This led us to believe that the model was not yet overfitting. This Hinton'ed us towards increasing the parameter capacity of the network. We added two additional convolutional layers right after the existing convolutional layers, additionally we increased the number of filters per convolutional layer (see final architecture section).
## Augmentation
Data augmentation is a very important step to reduce overﬁtting. We used real-time data augmentation on the following transformations:
*  **Translation shift** - Uniform between -3 and 3 pixels
*  **Rotation** - Uniform between -12 and 12 degrees
*  **Flipping** - Bernoulli with p = 0.5
*  **HSV augmentation** -  Uniform factors: Hue between 0.9 and 1.1, Saturation and value between 0.8 and 1.2

We figured that for the rotation transformation it does not make sense to rotate with a larger angle, since the objects in the images likely will never be upside down or heavily tilted.

To speed things up, we ran this augmentation in parallel for different images in seperate processes. The augmentation would then take 3 seconds per epoch approximately, regardless of batch size.

*Example augmentations*
![](https://raw.githubusercontent.com/gzuidhof/cad/master/assignment8/augment1.png)
![](https://raw.githubusercontent.com/gzuidhof/cad/master/assignment8/augment2.png)
![](https://raw.githubusercontent.com/gzuidhof/cad/master/assignment8/augment3.png)
![](https://raw.githubusercontent.com/gzuidhof/cad/master/assignment8/augment4.png)



## Batch Normalization and Leaky ReLUs##
Applying batch normalization presents the current state of the art in network architecture design. We apply batch normalization after the max pooling layers, this allows for faster convergence and a decrease in overfitting.

We investigated using Leaky Rectified Linear Units (ReLU). We empiricially determined that it improved our accuracy slightly.

## Regularization ##
We applied a penalty to the weights to further prevent overfitting, L2 regularization to be exact. We empirically evaluated a few values for the lambda term (the weighting of this penalty), and after some observations we settled for 0.02 which is relatively low.

## Final Architecture ##
* **Convolution layer** - 32 filters, filter size 3x3, LReLU, weights initialized uniform Glorot
* **Convolution layer** - 64 filters, filter size 3x3, LReLU, weights initialized uniform Glorot
* **Max Pooling layer** - size 2x2
* **Batch normalization layer**
* **Convolution layer** - 128 filters, filter size 3x3, LReLU, weights initialized uniform Glorot
* **Convolution layer** - 128 filters, filter size 3x3, LReLU, weights initialized uniform Glorot
* **Max Pooling layer** - size 2x2
* **Batch normalization layer**

* *"Deeper" network only:*
  * **Convolution layer** - 128 filters, filter size 3x3, LReLU, weights initialized uniform Glorot
  * **Convolution layer** - 128 filters, filter size 3x3, LReLU, weights initialized uniform Glorot
  * **Batch normalization layer**

* **Fully connected layer** - 1024 units, LReLU, weights initialized uniform Glorot
* **Fully connected layer** - 512 units, LReLU, weights initialized uniform Glorot
* **Softmax layer** - 10 outputs, one for each class

## Results: ##
We arbitrarily defined the number of maximum epochs, and at that epoch calculated the performance on the *test* set (as opposed to the validation set). We only report the test set performance at this final epoch, the validation set performance can be seen in the graphs (red line).


** Network without non-leaky ReLUs and only 1 fully connected layer *(0.827 accuracy)***
![](https://raw.githubusercontent.com/gzuidhof/cad/master/assignment8/1dense_NoLeaky_827.png)

** Network described above *(0.846 accuracy)***
![](https://raw.githubusercontent.com/gzuidhof/cad/master/assignment8/2dense_Leaky846.png)

Note that for the deeper networks we increased the amount of epochs, so they may be hard to compare to the earlier architectures.

** "Deeper" network *(0.850 accuracy)* **
![](https://raw.githubusercontent.com/gzuidhof/cad/master/assignment8/2dense_Leaky_Deeper_850.png)

** "Deeper" network ith 0.1 regularization lambda (as opposed to 0.02) *(0.852 accuracy)***
![](https://raw.githubusercontent.com/gzuidhof/cad/master/assignment8/2dense_leaky_010regularization_852.png)

Some more attempts were made (see GitHub repository for graphs), but they were mostly stopped before convergence.

## Let's go even deeper ##

Out of curiosity, we trained one final different (deeper) architecture with quite some changes, such as also doing zoom augmentations (0.90 to 1.1 scale factor) and bigger ranges of saturation and value augmentations (-0.25, 0.25). Also, we used RMSProp, as something funky happened around epoch 50 which caused the weights to explode that we could not explain. Finally, the batch size was increased to 256 and it features 1x1 convolutions.

The architecture:

* **Convolution layer** - 32 filters, filter size 3x3, LReLU, weights initialized uniform Glorot
* **Convolution layer** - 64 filters, filter size 3x3, LReLU, weights initialized uniform Glorot
* **Max Pooling layer** - size 2x2
* **Batch normalization layer**
* **Convolution layer** - 96 filters, filter size 3x3, LReLU, weights initialized uniform Glorot
* **Convolution layer** - 128 filters, filter size 1x1, LReLU, weights initialized uniform Glorot
* **Max Pooling layer** - size 2x2
* **Batch normalization layer**
* **Convolution layer** - 128 filters, filter size 3x3, LReLU, weights initialized uniform Glorot
* **Convolution layer** - 128 filters, filter size 1x1, LReLU, weights initialized uniform Glorot
* **Batch normalization layer**
* **Fully connected layer** - 1024 units, LReLU, weights initialized uniform Glorot
* **Fully connected layer** - 1024 units, LReLU, weights initialized uniform Glorot
* **Softmax layer** - 10 outputs, one for each class

We let it run for 100 epochs.. causing it to overfit quite a bit. The highest observed validation score of this network however, was 0.870.

** "Alternative network *(0.844 accuracy)***
![](https://raw.githubusercontent.com/gzuidhof/cad/master/assignment8/wemustgodeeper.png)


An easy way to improve the performance of our networks is by performing test time augmentation (TTA). Another obvious way is ensembling multiple networks and applying better regularization (our selected lambda values are likely not optimal).
