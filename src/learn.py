from __future__ import division
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import features
import numpy as np
import util
import dataset
from tqdm import tqdm

def train(X_train, X_test, y_train, y_test,clf, use_probability=True, predict_black=False):

    #print np.array(X_train).shape
    #print y_train.shape
    print "Fitting model on {} points".format(len(X_train))
    clf = clf.fit(X_train, y_train)

    print "Done, now predicting.."
    predictions = []

    if predict_black:
        if use_probability:
            predictions = clf.predict_proba(X_test)[:,1]
        else:
            predictions = clf.predict(X_test)
    else: #The fully black pixels are not predicted, much faster for SVM
        predictions = []
        for test_sample in tqdm(X_test):
            if sum(test_sample) == 0:
                p = 0
            elif use_probability:
                p = clf.predict_proba(test_sample.reshape(1,-1))[0][1]
            else:
                p = clf.predict(test_sample.reshape(1,-1))[0]

            predictions.append(p)

        predictions = np.array(predictions)

    out = predictions
    predict_fully_black(X_test, y_test, out)


    decision_boundary = 0.5
    out_binary = np.where(out > decision_boundary, 1,0)

    #Split the pixels into the original images again
    out_images_binary = util.chunks(out_binary, 384*512)
    out_images = util.chunks(out,384*512)
    y_images = util.chunks(y_test, 384*512)

    print "Done, calculating Dice..."
    dice(out_images_binary, y_images)

    print "Done, showing predicted images.."

    for image in out_images:
        end_image = image[:].reshape((512,384))
        true_pred = end_image >= 0.5
        prediction_thresholded = np.zeros(end_image.shape)
        prediction_thresholded[true_pred] = 1

        print np.mean(end_image)
        dataset.show_image([end_image,prediction_thresholded])

    print "Done."

def predict_fully_black(X_test, y_test, predictions):

    feature_sums = np.sum(X_test, axis=1)
    indices_fully_black = np.where(feature_sums == 0)[0]

    #Set the fully black places to zero in the predictions
    predictions[indices_fully_black] = 0

def dice(prediction, y):
    print "Calculating dice score"
    dices = [dice_score_img(p,t) for p,t in tqdm(zip(prediction,y))]
    print "Dice score mean {0}, std: {1}".format(np.mean(dices), np.std(dices))

def dice_score_img(p, y):
    return np.sum(p[y == 1]) * 2.0 / (np.sum(p) + np.sum(y))


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

    #train(X_train, X_test, y_train, y_test,LogisticRegression(), predict_black=True)
    train(X_train, X_test, y_train, y_test,RandomForestClassifier(n_estimators=10,n_jobs=-1), use_probability=True, predict_black=True)
    #train(X_train, X_test, y_train, y_test,SVC(verbose=True,max_iter=50000), use_probability=False)
