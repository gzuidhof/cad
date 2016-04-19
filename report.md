

### Assignment week 8, Using Convolutional Neural Networks for classification of CIFAR-10 dataset

Guido Zuidhof (s4160703), Steven Reitsma (s4132343), Robbert van der Gugten (s4137140).  
15/4/2016

----
## Implementation
We refer to our [Git repository](https://github.com/gzuidhof/cad) for our source code. We make use of the following dependencies:
* **Python 2.7**
* **OpenCV2**: morphological editing functions, used a lot in preprocessing
* **cPickle**: read and write files
* **tqdm**: progress bars
* **theano & lasagne**: framework for CNNs

## Initial network
Our network's architecture:
* **Convolution layer** - 32 filters, filter size 5x5, ReLU, weights initiliazed uniform Glorot
* **Max Pooling layer** - filter size 2x2
* **Convolution layer** - 64 filters, filter size 3x3, ReLU, weights initiliazed uniform Glorot
* **Max Pooling layer** - filter size 2x2
* **Fully connected layer** - 1024 units, ReLU, weights initiliazed uniform Glorot
* **Softmax layer** - 10 outputs, one for each class

This architecture resulted in a validation accuracy of 66% and a test accuracy of **TODO**


## Learning rate optimization & Choosing the update rule
Our initial learning rate is 0.01. We set the batch size to 64 resulting in less noisy gradients. We chose Stochastic Gradient Descent as the initial update rule. We improved this by adding momentum for faster convergence. If the learning rate is too high the optimizer will overshoot local optima. If the learning rate is too low, the optimization process will be slow. We tested learning rates of 0.1, 0.01 and 0.001 on an intermediate network to illustrate this. After 20 epochs a learning rate of 0.1 resulted in an accuracy of 73.1%, a learning rate of 0.01 resulted in 80.7% and a learning rate of 0.001 achieved 75.2%. We tried the update rule ADAM (Adaptive Moment Estimation), which optimizes the learning rate for each parameter separately. This did not result in a better accuracy (70.2% test accuracy as opposed to the sgd's 76%) for the simple network with data augmentation and batch normalization.

## Improving the architecture
We observed that the difference between our validation loss and training loss was small. This led us to believe that the model was not yet overfitting. This Hinton'ed us towards increasing the parameter capacity of the network. We added two additional convolutional layers right after the existing convolutional layers, additionally we increased the number of filters per convolutional layer (see final architecture section).
## Augmentation
Data augmentation is a very important step to reduce overﬁtting. We used real-time data augmentation on the following transformations:
*  **Translation shift** - Uniform between -3 and 3 pixels
*  **Rotation** - Uniform between -12 and 12 degrees
*  **Flipping** - Bernoulli with p = 0.5
*  **HSV augmentation** -  Uniform factors: Hue between 0.9 and 1.1, Saturation and value between 0.8 and 1.2

We figured that for the rotation transformation it does not make sense to rotate with a larger angle, since the objects in the images likely will never be upside down or heavily tilted.

## Batch Normalization ##
Applying batch normalization presents the current state of the art in network architecture design. We apply batch normalization after the max pooling layers, this allows for faster convergence and a decrease in overfitting.

## Regularization ##
We applied a penalty to the weights to further prevent overfitting, L2 regularization to be exact. We empirically evaluated a few values for the lambda term (the weighting of this penalty), and after some observations we settled for 0.02 which is relatively low.

## Final Architecture ##
We investigated using Leaky Rectified Linear Units (ReLU). This improved ? our accuracy slightly. 
```python
  def dist_transform_feature(image, threshold):
      kernel = np.ones((3,3),np.uint8)

      #Create a binary threshold for the image
      mask = np.array(np.where(image >= threshold, 1,0), dtype=np.uint8)

      #Closing operation
      closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

      #Calculate distance transform
      distance_transform = cv2.distanceTransform(closing, cv2.cv.CV_DIST_L2,5)

      return distance_transform
```




