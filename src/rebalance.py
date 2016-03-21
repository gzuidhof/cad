import numpy as np
import features
from tqdm import tqdm

# Class rebalancing

def balance_classes(X_train,y_train, positive_ratio=0.10):

    positives = [x for x,y in enumerate(y_train) if y==1]
    positives_y = y_train[positives]
    positives_x = X_train[positives]

    negatives = [x for x,y in enumerate(y_train) if y==0]
    negatives_y = y_train[negatives]
    negatives_x = X_train[negatives]

    n_negative = int (((1/positive_ratio) - 1) * len(positives))
    n_negative = min(len(negatives), n_negative)

    print "{0} positive, {1} negative before balancing".format(len(positives), len(negatives))
    print "{0} desired positive ratio, keeping {1} negative".format(positive_ratio, n_negative)

    indices = np.random.choice(len(negatives),n_negative,replace=False)
    return  np.concatenate((positives_x,negatives_x[indices])),np.concatenate((positives_y,negatives_y[indices]))

def remove_completely_black(X_train, y_train):
    feature_sums = np.sum(X_train, axis=1)
    print feature_sums

    #Not fully black indices
    indices_to_keep = np.where(feature_sums > 0)[0]

    print "N feature vectors {0}, not fully black {1}".format(len(X_train),len(indices_to_keep))

    return X_train[indices_to_keep], y_train[indices_to_keep]

def normalize_features(X_train, X_test):
    n_features = X_train.shape[1]

    feature_sums = np.sum(X_test, axis=1)
    nonblack_vectors = np.where(feature_sums > 0,1,0)
    #print nonblack_vectors.shape

    mask = []
    for x in range(X_test.shape[0]):
        mask.append([nonblack_vectors[x]]*n_features)
    mask = np.array(mask)

    X_test_nonblack = X_test[np.where(feature_sums > 0)]

    X = np.concatenate((X_train, X_test_nonblack))
    print X, X.shape

    mean = np.mean(X,axis=0)
    std = np.std(X,axis=0)

    for d in tqdm(range(len(X_train))):
        X_train[d] = (X_train[d] - mean) / std
    for d in tqdm(range(len(X_test))):
        X_test[d] = (X_test[d] - mean) / std

    #Make once fully black vectors fully black again
    X_test = X_test*mask

    return X_train, X_test



if __name__ == "__main__":
    print "Loading X"
    X_train, X_test = features.load_features()
    print "Loading Y"
    y_train, y_test = features.load_y()

    print "Removing fully black features"
    X_train, y_train = remove_completely_black(X_train, y_train)

    print "Normalizing features"
    normalize_features(X_train, X_test)

    print "Balancing classes"
    X_train,y_train = balance_classes(X_train,y_train)



    features.write_features((X_train, X_test),"balanced")
    features.write_y((y_train, y_test),"balanced")
