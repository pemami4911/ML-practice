#!/usr/bin/env python
from __future__ import division

import cPickle
import argparse
import os
import pandas
import numpy as np


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


def preprocess(path):
    df = pandas.read_csv(path, delimiter=';')

    # age

    # throw out data for age values > 60
    df = df.drop(df[df.age > 60].index)
    # normalize all values between -1 and +1
    df["age"] = min_max_scaling(df["age"], -1., 1.)

    # job
    df["job"] = df["job"].astype('category')
    df["job"] = df["job"].cat.codes
    df["job"] = min_max_scaling(df["job"], -1., 1.)

    # marital
    df["marital"] = df["marital"].astype('category')
    df["marital"] = df["marital"].cat.codes
    df["marital"] = min_max_scaling(df["marital"], -1., 1.)

    # balance
    df["balance"] = min_max_scaling(df["balance"], -1., 1.)

    # day
    df["day"] = min_max_scaling(df["day"], -1., 1.)

    # duration
    df["duration"] = min_max_scaling(df["duration"], -1., 1.)

    # campaign
    df["campaign"] = min_max_scaling(df["campaign"], -1., 1.)

    # previous
    df["previous"] = min_max_scaling(df["previous"], -1., 1.)

    # education
    df["education"] = df["education"].astype('category')
    df["education"] = df["education"].cat.codes
    df["education"] = min_max_scaling(df["education"], -1., 1.)

    # contact
    df["contact"] = df["contact"].astype('category')
    df["contact"] = df["contact"].cat.codes
    df["contact"] = min_max_scaling(df["contact"], -1., 1.)

    # month
    df["month"] = df["month"].astype('category')
    df["month"] = df["month"].cat.codes
    df["month"] = min_max_scaling(df["month"], -1., 1.)

    # poutcome
    df["poutcome"] = df["poutcome"].astype('category')
    df["poutcome"] = df["poutcome"].cat.codes
    df["poutcome"] = min_max_scaling(df["poutcome"], -1., 1.)

    # drop pdays, too much missing data
    df.drop('pdays', axis=1)

    # binary variables
    df["default"] = df["default"].astype('category')
    df["default"] = df["default"].cat.codes
    df["default"] = min_max_scaling(df["default"], -1., 1.)

    # housing
    df["housing"] = df["housing"].astype('category')
    df["housing"] = df["housing"].cat.codes
    df["housing"] = min_max_scaling(df["housing"], -1., 1.)

    # loan
    df["loan"] = df["loan"].astype('category')
    df["loan"] = df["loan"].cat.codes
    df["loan"] = min_max_scaling(df["loan"], -1., 1.)

    # y
    df["y"] = df["y"].astype('category')
    df["y"] = df["y"].cat.codes
    df["y"] = min_max_scaling(df["y"], -1., 1.)

    # split into train and test set
    # shuffle data
    df = df.iloc[np.random.permutation(len(df))]
    data = np.array_split(df, 2)
    train_data = data[0].values
    test_data = data[1].values

    save_pkl(train_data, args['training_data'])
    save_pkl(test_data, args['test_data'])

    return train_data, test_data

def sign(val):
    return 1. if val >= 0 else -1.

def pla(args, train_data, test_data):
    """

    :param args:
    :param data: data[0] contains the feature vectors, data[1] contains the labels
    :param params:
    :return: params:
    """
    rows, cols = np.shape(train_data)

    params = np.zeros((cols-2, 1), dtype=np.float32)

    xs = train_data[:, 0:cols-2]
    ys = train_data[:, cols-1]

    for i in range(args['n_epochs']):
        for j in range(rows):
            out = sign(np.dot(xs[j, :], params))
            if not ys[j] == out:
                params = np.add(params, np.reshape(np.multiply(ys[j], xs[j, :]), (cols-2, 1)))

        # re-normalize the parameter vector so ||theta|| = 1
        params = np.divide(params, np.linalg.norm(params))

        err = test(test_data, params)

        print("  [*] PLA test error of {} % at epoch {}...".format(np.round(err * 100, 4), i))

    return params


def test(test_data, params):

    rows, cols = np.shape(test_data)

    xs = test_data[:, 0:cols-2]
    ys = test_data[:, cols-1]

    wrong = 0
    for i in range(rows):
        out = sign(np.dot(xs[i, :], params))
        if not ys[i] == out:
            wrong += 1

    return (wrong / rows)

def test_baseline(test_data):
    """Guess no every time."""

    rows, cols = np.shape(test_data)

    xs = test_data[:, 0:cols - 2]
    ys = test_data[:, cols - 1]

    wrong = 0
    for i in range(rows):
        if not ys[i] == 1.0:
            wrong += 1

    return (wrong / rows)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Set the run parameters.')
    parser.add_argument('--dataset', default='data/bank.csv', help='raw CSV file of data')
    parser.add_argument('--training_data', default='data/bank_train.pkl', help='specify pickled preprocessed training data')
    parser.add_argument('--test_data', default='data/bank_test.pkl', help='specify pickled preprocessed test data')
    parser.add_argument('--n_epochs', default=1, help='num epochs of training')
    parser.add_argument('--random_seed', default=1234)

    args = vars(parser.parse_args())

    np.random.seed(args['random_seed'])

    if os.path.isfile(os.path.join(os.getcwd(), args['training_data'])) and \
            os.path.isfile(os.path.join(os.getcwd(), args['test_data'])):
        train_data = load_pkl(args['training_data'])
        test_data = load_pkl(args['test_data'])
    else:
        train_data, test_data = preprocess(args['dataset'])

    params = pla(args, train_data, test_data)

    baseline_err = test_baseline(test_data)

    print("  [*] baseline test error is {} %...".format(np.round(baseline_err * 100, 3)))
    