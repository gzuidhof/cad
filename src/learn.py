from __future__ import division
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import features
import numpy as np
import util
import dataset
from tqdm import tqdm
import scipy.optimize

def train(X_train, X_test, y_train, y_test,clf, use_probability=True, predict_black=False,name="NoName"):

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

    #Threshold optimization
    decision_boundary = threshold_optimization(out, y_test)

    #Apply threshold
    out_binary = np.where(out > decision_boundary, 1,0)

    #Split the pixels into the original images again
    out_images_binary = util.chunks(out_binary, 384*512)
    out_images = util.chunks(out,384*512)
    y_images = util.chunks(y_test, 384*512)

    print "Done, calculating Dice..."

    mean, std, dices = dice(out_images_binary, y_images)
    print "Dice score mean {0}, std: {1}".format(mean, std)

    joblib.dump(dices, dataset.DATA_FOLDER+name+"_dice.pkl")

    print "Done, showing predicted images.."

    for image in out_images:
        end_image = image[:].reshape((512,384))
        true_pred = end_image >= 0.5
        prediction_thresholded = np.zeros(end_image.shape)
        prediction_thresholded[true_pred] = 1

        print np.mean(end_image)
        dataset.show_image([end_image,prediction_thresholded])

    print "Done."

def threshold_optimization(p, y):
    print "Optimizing threshold"
    y_images = util.chunks(y, 384*512)

    def dice_objective(threshold):
        p_binary = np.where(p > threshold, 1,0)
        p_images_binary = util.chunks(p_binary, 384*512)

        mean, std, dices = dice(p_images_binary, y_images)
        return -mean

    x, v, message = scipy.optimize.fmin_l_bfgs_b(dice_objective, 0.5, approx_grad=True, bounds=[(0, 1)], epsilon=1e-03)
    print "Optimized, threshold {0}, ? {1}, termination because {2}".format(x,v,message)
    return x[0]

def threshold_optimization_naive(p,y):
    print "Optimizing threshold"
    y_images = util.chunks(y, 384*512)

    candidates = np.arange(0.25,0.75,1/2500)

    def dice_objective(threshold):
        p_binary = np.where(p > threshold, 1,0)
        p_images_binary = util.chunks(p_binary, 384*512)

        mean, std, dices = dice(p_images_binary, y_images)
        return mean

    #score = map(dice_objective,tqdm(candidates))
    scores = []
    for t in tqdm(candidates):
        score = dice_objective(t)
        scores.append(score)
    print np.argmax(scores)
    threshold = candidates[np.argmax(scores)]
    print "Best threshold ", threshold
    return threshold

def predict_fully_black(X_test, y_test, predictions):

    feature_sums = np.sum(X_test, axis=1)
    indices_fully_black = np.where(feature_sums == 0)[0]

    #Set the fully black places to zero in the predictions
    predictions[indices_fully_black] = 0

def dice(prediction, y):
    dices = [dice_score_img(p,t) for p,t in zip(prediction,y)]
    mean = np.mean(dices)
    std = np.std(dices)

    return mean, std, dices

def dice_score_img(p, y):
    return np.sum(p[y == 1]) * 2.0 / (np.sum(p) + np.sum(y))


def features_to_images(features, dim=0):
    images = util.chunks(features,384*512)
    for im in images:
        end_image = im[:,dim].reshape((512,384))
        print np.mean(end_image)

if __name__ == "__main__":
    print "\nLoading X"
    X_train, X_test = features.load_features("balanced")

    print "Loading Y"
    y_train, y_test = features.load_y("balanced")

    #train(X_train, X_test, y_train, y_test,LogisticRegression(), predict_black=True,name="logreg")
    train(X_train, X_test, y_train, y_test,AdaBoostClassifier(n_estimators=200,random_state=42), predict_black=True,name="adaboost200")
    #train(X_train, X_test, y_train, y_test,RandomForestClassifier(n_estimators=250,n_jobs=-1,random_state=42), use_probability=True, predict_black=True,name="rf200")
    #train(X_train, X_test, y_train, y_test,SVC(verbose=2,max_iter=100000,probability=True), use_probability=True,name="svmrbf")
    #train(X_train, X_test, y_train, y_test,SVC(kernel="linear",verbose=2,max_iter=100000,probability=True), use_probability=True,name="svmlinear")
