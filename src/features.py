import numpy as np
import dataset
import cPickle as pickle

def write_features(features):
    with open(dataset.DATA_FOLDER+"features.p", 'w') as f:
        pickle.dump(features, f)
        
def write_y(y):
    with open(dataset.DATA_FOLDER+"y.p", 'w') as f:
        pickle.dump(y, f) 
        

def flatten_images(dataset):
    n_images = dataset.shape[0]
    return dataset.reshape((n_images,-1))


def get_features(x):
    x1, x2, x3 = x
    x1 = np.array(x1).reshape((512*384))
    x2 = np.array(x2).reshape((512*384))
    x3 = np.array(x3).reshape((512*384))
        
        
    dat = np.zeros((x1.shape[0], 3))
    for n in range(len(x)):
        dat[n,:] = [x1[n],x2[n],x3[n]]
       
    return dat
    
def load_features():
    with open(dataset.DATA_FOLDER+"features.p", 'r') as f:
        return np.array(pickle.load(f))
        
def load_y():
    with open(dataset.DATA_FOLDER+"y.p", 'r') as f:
        return np.array(pickle.load(f))

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = dataset.load_dataset()
    
    features_train = np.array([get_features(x) for x in X_train])
    features_test = np.array([get_features(x) for x in X_test])
    
    y_train = flatten_images(np.array(y_train))
    y_test = flatten_images(np.array(y_test))
    
    
    #print features_train.shape
    #features_train = features_train.flatten()
    #features_test = features_test.flatten()
    
    features_train = np.concatenate(features_train)
    features_test = np.concatenate(features_test)
    print features_train.shape
    print features_test.shape
    
    
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    print "Writing X to file"
    #write_features((features_train, features_test))
    write_features((features_train,features_test))
    print "Writing Y to file"
    write_y((y_train, y_test))
    print "Done"
    
    
    
    
    
    
    
    