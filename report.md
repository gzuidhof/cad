

### Assignment week 8, Using Convolutional Neural Networks for classification of CIFAR-10 dataset

Guido Zuidhof (s4160703), Robbert van der Gugten (s4137140).  
15/4/2016

----
## Implementation
We refer to our [Git repository](https://github.com/gzuidhof/cad) for our source code. We make use of the following dependencies:
* **Python 2.7**
* **OpenCV2**: morphological editing functions, used a lot in preprocessing
* **cPickle**: read and write files
* **tqdm**: progress bars
* **theano & lasagne**: framework for CNNs

## I nitial network
Our network's architecture:
* **Convolution layer** - 32 filters, filter size 5x5, ReLU, weights initiliazed uniform Glorot
* **Max Pooling layer** - filter size 2x2
* **Convolution layer** - 64 filters, filter size 3x3, ReLU, weights initiliazed uniform Glorot
* **Max Pooling layer** - filter size 2x2
* **Fully connected layer** - 1024 units, ReLU, weights initiliazed uniform Glorot
* **Softmax layer** - 10 outputs, one for each class

This architecture resulted in a validation accuracy of 66% and a test accuracy of **TODO**


## Learning rate optimization

## Choosing the update rule

## Improving the architecture

## Augmentation



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




