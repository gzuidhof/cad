import numpy as np
import os
import pickle

dataset_dir = "../data"

def load():

    # training set, batches 1-4
    train_X = np.zeros((40000, 3, 32, 32), dtype="float32")
    train_y = np.zeros((40000, 1), dtype="ubyte").flatten()
    n_samples = 10000 # number of samples per batch
    for i in range(0,4):
        f = open(os.path.join(dataset_dir, "data_batch_"+str(i+1)+""), "rb")
        cifar_batch = pickle.load(f)
        f.close()
        train_X[i*n_samples:(i+1)*n_samples] = (cifar_batch['data'].reshape(-1, 3, 32, 32) / 255.).astype("float32")
        train_y[i*n_samples:(i+1)*n_samples] = np.array(cifar_batch['labels'], dtype='ubyte')

    # validation set, batch 5
    f = open(os.path.join(dataset_dir, "data_batch_5"), "rb")
    cifar_batch_5 = pickle.load(f)
    f.close()
    val_X = (cifar_batch_5['data'].reshape(-1, 3, 32, 32) / 255.).astype("float32")
    val_y = np.array(cifar_batch_5['labels'], dtype='ubyte')

    # labels
    f = open(os.path.join(dataset_dir, "batches.meta"), "rb")
    cifar_dict = pickle.load(f)
    label_to_names = {k:v for k, v in zip(range(10), cifar_dict['label_names'])}
    f.close()

    print("training set size: data = {}, labels = {}".format(train_X.shape, train_y.shape))
    print("validation set size: data = {}, labels = {}".format(val_X.shape, val_y.shape))

    return train_X, train_y, val_X, val_y, label_to_names


