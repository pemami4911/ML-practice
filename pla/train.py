#!/usr/bin/env python

import cPickle
import argparse
import os
import pandas


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

    # throw out data for age values > 60
    df = df.drop(df[df.age > 60].index)
    # normalize all values between -1 and +1
    df["age"] = min_max_scaling(df["age"], -1., 1.)
    print(df["age"])
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Set the run parameters.')
    parser.add_argument('--dataset', default='data/bank.csv', help='raw CSV file of data')
    parser.add_argument('--training_data', default='data/bank_train.pkl', help='specify pickled preprocessed training data')
    parser.add_argument('--test_data', default='data/bank_test.pkl', help='specify pickled preprocessed test data')
    parser.add_argument('--n_epochs', default=10, help='num epochs of training')

    args = vars(parser.parse_args())

    if os.path.isfile(args['training_data']) and os.path.isfile(args['test_data']):
        train_data = load_pkl(args['training_data'])
        test_data = load_pkl(args['test_data'])
    else:
        train_data, test_data = preprocess(args['dataset'])
