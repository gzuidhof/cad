from __future__ import division
import numpy as np
import sklearn.cross_validation
import glob
import scipy.misc
import matplotlib.pyplot as plt
import cPickle as pickle
import tqdm

DATA_FOLDER = "../data/"
IMAGES_FOLDER = "../data/images/"


def load_dataset():
    with open(DATA_FOLDER+"dataset.p", 'r') as f:
        dataset = pickle.load(f)

    X_train, X_test, y_train, y_test = dataset
    return X_train, X_test, y_train, y_test

def load_images(filenames):
    data = []
    for filename in filenames:
        im = scipy.misc.imread(filename)
        data.append(im)

    return data

def show_image(im):
    plt.figure()
    plt.imshow(im, cmap='Greys_r')
    plt.show()

if __name__ == "__main__":

    print "Loading images"
    y = load_images(glob.glob(IMAGES_FOLDER+"*an.png"))
    x1 = load_images(glob.glob(IMAGES_FOLDER+"*t1.png"))
    x2 = load_images(glob.glob(IMAGES_FOLDER+"*t2.png"))
    x3 = load_images(glob.glob(IMAGES_FOLDER+"*fl.png"))

    # Make the annotations fully binary
    for im in y:
        im[im<=150] = 0
        im[im>150] = 1

    # Make the pixel values run from 0 to 1
    x1 = [x/255 for x in x1]
    x2 = [x/255 for x in x2]
    x3 = [x/255 for x in x3]

    X = zip(x1,x2,x3)

    print "Splitting dataset into train, test"
    X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X,y, test_size=0.33, random_state=0)

    dataset = (X_train, X_test, y_train, y_test)

    print "Writing to file"
    with open(DATA_FOLDER+"dataset.p", 'w') as f:
        pickle.dump(dataset, f)

    print "Done."
