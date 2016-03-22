from __future__ import division
import numpy as np
import dataset
import hickle as pickle
from tqdm import tqdm
from skimage.feature import blob_dog, blob_log, blob_doh
import cv2

def write_features(features, name=""):
    with open(dataset.DATA_FOLDER+"features_{0}.p".format(name), 'w') as f:
        pickle.dump(features, f)

def write_y(y,name=""):
    with open(dataset.DATA_FOLDER+"y_{0}.p".format(name), 'w') as f:
        pickle.dump(y, f)


def flatten_images(dataset):
    n_images = dataset.shape[0]
    return dataset.reshape((n_images,-1))


#Determine the features for one scan (3 input images: T1, T2, Flair)
def get_features(x):
    #Amount of pixels
    N = 512*384

    x1, x2, x3 = x
    x1 = np.array(x1)
    x2 = np.array(x2)
    x3 = np.array(x3)

    #Determine other features

    #Distance transform to edge
    x4 = dist_transform_feature(x1)
    #Distance to ventricles / folds
    x5 = dist_transform_feature(x1,100)

    #Laplacian of Gaussian blob features
    x9  = blob_feature(x1, 'log')
    x10 = blob_feature(x2, 'log')
    x11 = blob_feature(x3, 'log')

    x12  = blob_feature(x1, 'doh')
    x13 = blob_feature(x2, 'doh')
    x14 = blob_feature(x3, 'doh')


    #EquaLized images
    x6 = histogram_equalization(x1)
    x7 = histogram_equalization(x2)
    x8 = histogram_equalization(x3)

    #Show features
    #dataset.show_image([np.vstack((x1,x3)), np.vstack((x2,x4))])

    #This could use a rewrite ;)

    features = [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14]

    #Flatten features
    features = [f.reshape((N,)) for f in features]

    dat = []
    for n in range(N):

        dat.append([f[n] for f in features] )

    dat = np.array(dat)

    return dat

def histogram_equalization(image, adaptive=False):
    if adaptive:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        image = clahe.apply(image)
    else:
        image = cv2.equalizeHist(image)
    return image

def dist_transform_feature(image, threshold=1):
    kernel = np.ones((3,3),np.uint8)
    mask = np.array(np.where(image >= threshold, 1,0), dtype=np.uint8)
    #mask = np.minimum(image, 1)

    #Closing operation
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    #Show result of closing
    #dataset.show_image(closing-mask)

    #Calculate distance transform
    distance_transform = cv2.distanceTransform(closing, cv2.cv.CV_DIST_L2,5)
    #dataset.show_image(distance_transform)

    return distance_transform

def blob_feature(image, method='log'):
    if method == 'log':
        blobs = blob_log(image, )
    else:
        blobs = blob_doh(image, )

    blob_image = np.zeros(image.shape)

    #Draw the blobs to an image
    for blob in blobs:
        y,x,sigma = blob
        color = sigma
        size = int(np.sqrt(2*sigma)) if method == 'log' else sigma
        cv2.circle(blob_image, (x, y), size, sigma/1,-1)

    #dataset.show_image(blob_image)
    return blob_image



def load_features(name=""):
    with open(dataset.DATA_FOLDER+"features_{0}.p".format(name), 'r') as f:
        return np.array(pickle.load(f))

def load_y(name=""):
    with open(dataset.DATA_FOLDER+"y_{0}.p".format(name), 'r') as f:
        return np.array(pickle.load(f))

def generate_features_main():
    print "Loading dataset"
    X_train, X_test, y_train, y_test = dataset.load_dataset()

    print "Creating features of train set ({} images)".format(len(X_train))
    features_train = np.array([get_features(x) for x in tqdm(X_train)])

    print "Creating features of test set  ({} images)".format(len(X_test))
    features_test = np.array([get_features(x) for x in tqdm(X_test)])

    print "Flattening individual annotated images"
    y_train = flatten_images(np.array(y_train))
    y_test = flatten_images(np.array(y_test))

    print "Flattening all pixels"
    features_train = np.concatenate(features_train)
    features_test = np.concatenate(features_test)

    y_train = y_train.flatten()
    y_test = y_test.flatten()

    print "Features train {0}, Features test {1}".format(features_train.shape, features_test.shape)
    print "Labels train {0}, Labels test {1}".format(y_train.shape, y_test.shape)

    print "Writing X to file"
    write_features((features_train,features_test))

    print "Writing Y to file"
    write_y((y_train, y_test))
    print "Done."

if __name__ == "__main__":
    generate_features_main()

    #im = dataset.load_images(["../data/images/1230931003_fl.png"])[0]
    #blobbed = blob_feature(im)
    #dataset.show_image([im,blobbed])
