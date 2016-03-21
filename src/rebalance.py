import numpy as np
import features

# Class rebalancing

def balance_classes(X_train,y_train, positive_ratio=0.05):

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


if __name__ == "__main__":
    print "Loading X"
    X_train, X_test = features.load_features()
    print "Loading Y"
    y_train, y_test = features.load_y()

    print "Removing fully black features"
    X_train, y_train = remove_completely_black(X_train, y_train)

    print "Balancing classes"
    X_train,y_train = balance_classes(X_train,y_train)

    features.write_features((X_train, X_test),"balanced")
    features.write_y((y_train, y_test),"balanced")
