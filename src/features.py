import numpy as np
import dataset
import cPickle as pickle
from tqdm import tqdm

def write_features(features):
    with open(dataset.DATA_FOLDER+"features.p", 'w') as f:
        pickle.dump(features, f)

def write_y(y):
    with open(dataset.DATA_FOLDER+"y.p", 'w') as f:
        pickle.dump(y, f)


def flatten_images(dataset):
    n_images = dataset.shape[0]
    return dataset.reshape((n_images,-1))


#Determine the features for one scan (3 input images: T1, T2, Flair)
def get_features(x):
    x1, x2, x3 = x
    x1 = np.array(x1).reshape((512*384))
    x2 = np.array(x2).reshape((512*384))
    x3 = np.array(x3).reshape((512*384))

    dat = []
    for n in range(512*384):
        dat.append([x1[n],x2[n],x3[n]] )

    dat = np.array(dat)

    return dat

def load_features():
    with open(dataset.DATA_FOLDER+"features.p", 'r') as f:
        return np.array(pickle.load(f))

def load_y():
    with open(dataset.DATA_FOLDER+"y.p", 'r') as f:
        return np.array(pickle.load(f))

if __name__ == "__main__":
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
