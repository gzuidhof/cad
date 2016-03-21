from __future__ import division
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import features
import numpy as np
import util
import dataset

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

        if len(image.shape) == 1:
            end_image = image[:].reshape((512,384))
        else:
            end_image = image[:,1].reshape((512,384))

        print np.mean(end_image)
        dataset.show_image([end_image,np.where(end_image >= 0.5, 1, 0)])

    print "Done."

def features_to_images(features, dim=0):
    images = util.chunks(features,384*512)
    for im in images:
        end_image = im[:,dim].reshape((512,384))
        print np.mean(end_image)

if __name__ == "__main__":
    print "Loading X"
    X_train, X_test = features.load_features("balanced")

    print "Loading Y"
    y_train, y_test = features.load_y("balanced")

    #train(X_train, X_test, y_train, y_test,LogisticRegression())
    train(X_train, X_test, y_train, y_test,RandomForestClassifier(n_estimators=20,n_jobs=-1))
    #train(X_train, X_test, y_train, y_test,SVC(verbose=True,max_iter=10000))
