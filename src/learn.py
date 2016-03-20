from __future__ import division
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import features
import numpy as np
import util
import dataset

def balance_classes(X_train,y_train, positive_ratio=0.02):


    positives = [x for x,y in enumerate(y_train) if y==1]
    positives_y = y_train[positives]
    positives_x = X_train[positives]

    negatives = [x for x,y in enumerate(y_train) if y==0]
    negatives_y = y_train[negatives]
    negatives_x = X_train[negatives]

    n_negative = int (((1/positive_ratio) - 1) * len(positives))

    print "{0} positive, {1} negative before balancing".format(len(positives), len(negatives))
    print "{0} desired positive ratio, keeping {1} negative".format(positive_ratio, n_negative)

    indices = np.random.choice(len(negatives),n_negative,replace=False)
    return  np.concatenate((positives_x,negatives_x[indices])),np.concatenate((positives_y,negatives_y[indices]))

def train(X_train, X_test, y_train, y_test,clf):

    #print np.array(X_train).shape
    #print y_train.shape
    print "Fitting model on {} points".format(len(X_train))
    clf = clf.fit(X_train, y_train)

    print "Done, now predicting.."
    out = clf.predict_proba(X_test)

    print "Done, showing predicted images.."
    out_images = util.chunks(out,384*512)
    for image in out_images:
        end_image = image[:,1].reshape((512,384))
        print np.mean(end_image)
        dataset.show_image(end_image)

    print "Done."

def features_to_images(features, dim=0):
    images = util.chunks(features,384*512)
    for im in images:
        end_image = im[:,dim].reshape((512,384))
        print np.mean(end_image)

if __name__ == "__main__":
    print "Loading X"
    X_train, X_test = features.load_features()
    #print sum(np.max(X_train, axis=1))
    #features_to_images(X_train)
    print "Loading Y"
    y_train, y_test = features.load_y()

    print "Balancing classes"
    X_train,y_train = balance_classes(X_train,y_train)
    train(X_train, X_test, y_train, y_test,LogisticRegression())
