import numpy as np
import cPickle


def min_max_scaling(X, max, min):
    X_std = (X - X.min()) / (X.max() - X.min())
    return X_std * (max - min) + min

def save_pkl(obj, path):
    with open(path, 'w') as f:
        cPickle.dump(obj, f)
        print("  [*] save %s" % path)

def load_pkl(path):
    with open(path) as f:
        obj = cPickle.load(f)
    print("  [*] load %s" % path)
    return obj

def sign(val):
    return 1. if val >= 0 else -1.

def sigmoid(x):
    """
    Returns a value between 0 and 1
    """
    return np.exp(-np.logaddexp(0, -x))

def mse(y, y_hat):
    """ MSE """
    return np.multiply(0.5, np.power((y_hat - y), 2.))

def binarize(x, threshold):
    """
    Binarize the output of a sigmoid to be either +1 or -1
    if output is gte threshold.
    """ 
    return 1. if x >= threshold else -1. 