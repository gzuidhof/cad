from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import features
import numpy as np
import util
import dataset


def train(X_train, X_test, y_train, y_test,clf):

    print np.array(X_train).shape
    print y_train.shape
    
    clf = clf.fit(X_train, y_train)
    print "classifier fit, predicting.."
    out = clf.predict_proba(X_test)
    print "predicted, showing predicted images.."
    out_images = util.chunks(out,384*512)
    for image in out_images:
        end_image = image[:,0].reshape((384,512))
        print np.mean(end_image)
        dataset.show_image(end_image)

if __name__ == "__main__":
    print "Loading X"
    X_train, X_test = features.load_features()
    print max(X_train.any())
    print "Loading Y"
    y_train, y_test = features.load_y()
    print max(y_train)
    
    train(X_train, X_test, y_train, y_test,LogisticRegression())