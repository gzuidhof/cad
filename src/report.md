### Preprocessing
### Selection

### Feature Engineering
We used a 15-dimensional feature set.

* **Intensity features**, the pixel values of the T1, T2 and FLAIR-weighted images (3x).
* **Distance transforms** to brain edge and folds/ventricles (2x).
  * Brain edge distance transform obtained by first thresholding the T1 weighted above 1 intensity, performing a closing morphology operation with a 3x3 kernel and finally computing the distance from every white pixel to a black pixel (see code snippet below).
  * The distance to ventricles or folds was obtained by the same process, but with a threshold of 100.

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
* **Blobness measures**, both Laplacian of Gaussian (`skimage.feature.blob_log`) and Determinant of Hessian ( `skimage.feature.blob_doh`). For every pixel the value of this feature is set to the size of the blob the pixel is part of (6x).
* **Histogram equalized intensities**. Intensity values of T1, T2 and FLAIR-weighted images after histogram equalization (3x). We also applied contrast limited adaptive histogram equalization (CLAHE), but this did not improve the classification result and was thus omitted.
* **Fraction of max feature**. We found that often the lesions were the brightest pixels in FLAIR-weighted images. This feature is a sort of inverse distance measure for every pixel to this intensity value, dramatized by performing some arbitrary high power, we used 4 (see code snippet below).  

```python
def fraction_of_max_feature(image):
    im_max = np.max(image)
    return (image/im_max)**4
```

### Selection
We removed all points that had a completely empty feature vector (sum=0) from the train set, these are the dark points outside of the brain in the image. This removes around 4 million pixels out of 6.5 million. Furthermore, we remove negative cases to slightly rebalance the dataset. We experimented with different fractions of positive cases, and ended up with a 5% share of positive cases resulting in the best performance.

### Preprocessing
We normalize all features so that they have zero mean and unit-variance. So for every feature dimension `x:= (x-mean(x))/std(x)`. Here, we are careful not to touch the completely black pixels in the test set.

### Postprocessing (threshold optimization)
After predictions are made, we optimize the decision boundary. In other words, the probability threshold above which the point is classified as a white matter lesion. We use the `L-BFGS-B` algorithm found in `scipy.optimize` for this step, with the Dice similarity coefficient as the objective function.

### Classification
