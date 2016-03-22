import numpy as np
import dataset
import cPickle as pickle
from tqdm import tqdm
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
    x4 = dist_transform_feature(x1)

    x1 = x1.reshape((N,))
    x2 = x2.reshape((N,))
    x3 = x3.reshape((N,))
    x4 = x4.reshape((N,))

    dat = []
    for n in range(N):
        dat.append([x1[n],x2[n],x3[n],x4[n]] )
        #dat.append([x1[n],x4[n]] )

    dat = np.array(dat)

    return dat

def dist_transform_feature(image):
    kernel = np.ones((3,3),np.uint8)
    mask = np.minimum(image, 1)

    #Closing operation
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    #Show result of closing
    #dataset.show_image(closing-mask)

    #Calculate distance transform
    distance_transform = cv2.distanceTransform(closing, cv2.cv.CV_DIST_L2,5)
    #dataset.show_image(distance_transform)

    return distance_transform


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
    #dataset.show_image([im,im])
    #dist_transform_feature(im)
