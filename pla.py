#!/usr/bin/env python
from __future__ import division, absolute_import

import argparse
import os
import matplotlib.pyplot as plt
import pandas
from sklearn.linear_model import LogisticRegression

from ops import *


def preprocess(path):
    df = pandas.read_csv(path, delimiter=';')

    # age

    # throw out data for age values > 60
    # df = df.drop(df[df.age > 60].index)
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

    # split into 50/50 train and test set
    # shuffle data
    df = df.iloc[np.random.permutation(len(df))]
    data = np.array_split(df, 2)
    train_data = data[0].values
    test_data = data[1].values

    save_pkl(train_data, args['training_data'])
    save_pkl(test_data, args['test_data'])

    return train_data, test_data


def prep_data(train_data, test_data, augment=False):
    """

    :param train_data
    :param test_data
    :param augment: set to True if the xs should be augmented with a 
        col of 1's to account for biases
    """

    np.random.shuffle(train_data)
    np.random.shuffle(test_data)

    rows, cols = np.shape(train_data)
    xs = train_data[:, 0:cols - 2]
    ys = train_data[:, cols - 1]

    if augment:
        xs = np.concatenate((xs, np.ones((rows, 1))), axis=1)

    test_rows, test_cols = np.shape(test_data)

    test_xs = test_data[:, 0:test_cols - 2]
    test_ys = test_data[:, test_cols - 1]

    if augment:
        test_xs = np.concatenate((test_xs, np.ones((test_rows, 1))), axis=1)

    return {
        'train_rows': rows,
        'train_cols': cols,
        'train_xs': xs,
        'train_ys': ys,
        'test_rows': test_rows,
        'test_cols': test_cols,
        'test_xs': test_xs,
        'test_ys': test_ys
    }


def pla(args, train_data, test_data):
    """
    Perceptron Learning Algorithm 

    :param args:
    :param train_data:
    :param test_data:
    """

    data = prep_data(train_data, test_data)
    n_features = data['train_cols'] - 2

    params = np.zeros((n_features, 1), dtype=np.float32)
    err = np.zeros((int(args['n_epochs']), 1))

    for i in range(int(args['n_epochs'])):

        for j in range(data['train_rows']):
            out = sign(np.dot(data['train_xs'][j, :], params))
            if not data['train_ys'][j] == out:
                # params update is: params <= params + y * 
                params = np.add(params, np.reshape(np.multiply(data['train_ys'][j],
                                                               data['train_xs'][j, :]), (n_features, 1)))

        # re-normalize the parameter vector so ||theta|| = 1
        params = np.divide(params, np.linalg.norm(params))

        wrong = 0
        for j in range(data['test_rows']):
            out = sign(np.dot(data['test_xs'][j, :], params))
            if not data['test_ys'][j] == out:
                wrong += 1

        err[i] = np.round((wrong / data['test_rows']) * 100, 4)

        print("  [PLA] validation error of {} % at epoch {}".format(err[i], i))

        # Re-shuffle the data for the next epoch
        data = prep_data(train_data, test_data)

    return err


def sigmoid_net(args, train_data, test_data):
    """
    Use the sigmoid activation function instead of 
    sign function

    :param args:
    :param train_data:
    :param test_data:
    """
    data = prep_data(train_data, test_data, augment=True)

    n_features = data['train_cols'] - 1
    # Add an extra parameter for the bias terms
    # params = np.zeros((cols-1, 1), dtype=np.float32)
    params = np.random.normal(scale=1, size=(n_features, 1))
    grads = np.zeros_like(params)
    loss = np.zeros((int(args['n_epochs']), 1))
    err = np.zeros_like(loss)

    for i in range(int(args['n_epochs'])):

        grads[:] = 0

        for j in range(data['train_rows']):
            out = sigmoid(np.dot(data['train_xs'][j, :], params))
            pred = binarize(out, 0.5)
            loss[i] += mse(data['train_ys'][j], pred)

            # gradient computation
            for k in range(n_features):
                dE_dout = np.multiply(-1, pred - data['train_ys'][j])
                dout_dnet = np.multiply(out, 1. - out)
                dnet_dwk = data['train_xs'][j, k]
                grads[k] += np.multiply(dE_dout, np.multiply(dout_dnet, dnet_dwk))

        scaled_grads = np.divide(grads, data['train_rows'])
        loss[i] = np.divide(loss[i], data['train_rows'])

        print("  [Sigmoid] training loss {} at epoch {}".format(np.round(loss[i], 4), i))

        # apply gradients
        lr = float(args['learning_rate'])
        params += np.multiply(lr, scaled_grads)

        # compute validation misclassification error
        wrong = 0
        for j in range(data['test_rows']):
            out = sigmoid(np.dot(data['test_xs'][j, :], params))
            pred = binarize(out, 0.5)
            if not data['test_ys'][j] == pred:
                wrong += 1

        err[i] = np.round((wrong / data['test_rows']) * 100, 4)

        print("  [Sigmoid] test error of {} % at epoch {}".format(err[i], i))

        # Re-shuffle the data for the next epoch
        data = prep_data(train_data, test_data, augment=True)

    return loss, err


def fixed_baseline(test_data):
    """Guess no every time."""

    rows, cols = np.shape(test_data)

    xs = test_data[:, 0:cols - 2]
    ys = test_data[:, cols - 1]

    wrong = 0
    for i in range(rows):
        if not ys[i] == 1.0:
            wrong += 1

    return wrong / rows


def logistic_regression_baseline(train_data, test_data):
    # use default parameters
    lr = LogisticRegression()

    _, cols = np.shape(train_data)

    xs = train_data[:, 0:cols - 2]
    ys = train_data[:, cols - 1]

    _, test_cols = np.shape(test_data)

    test_xs = test_data[:, 0:test_cols - 2]
    test_ys = test_data[:, test_cols - 1]

    lr.fit(xs, ys)
    accuracy = lr.score(test_xs, test_ys)
    return 1. - accuracy


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Set the run parameters.')
    parser.add_argument('--random_seed', default=2223)
    parser.add_argument('--model', default='PLA')
    parser.add_argument('--dataset', default='data/bank.csv', help='raw CSV file of data')
    parser.add_argument('--training_data', default='data/bank_train.pkl',
                        help='specify pickled preprocessed training data')
    parser.add_argument('--test_data', default='data/bank_test.pkl', help='specify pickled preprocessed test data')
    parser.add_argument('--n_epochs', default=100, help='num epochs of training')
    parser.add_argument('--learning_rate', default=0.1, help='learning rate for gradient descent')
    parser.add_argument('--plots', action='store_true', default=False)

    args = vars(parser.parse_args())

    np.random.seed(int(args['random_seed']))

    if os.path.isfile(os.path.join(os.getcwd(), args['training_data'])) and \
            os.path.isfile(os.path.join(os.getcwd(), args['test_data'])):
        train_data = load_pkl(args['training_data'])
        test_data = load_pkl(args['test_data'])
    else:
        train_data, test_data = preprocess(args['dataset'])

    if args['model'] == 'PLA':
        err = pla(args, train_data, test_data)
        if args['plots']:
            plt.plot(err)
            plt.ylabel('test error')
            plt.xlabel('epoch')
            plt.show()

    elif args['model'] == 'sigmoid':
        loss, err = sigmoid_net(args, train_data, test_data)
        if args['plots']:
            plt.figure(1)
            plt.subplot(121)
            plt.plot(err)
            plt.ylabel('test error')
            plt.xlabel('epoch')
            plt.title('sigmoid net error')

            plt.subplot(122)
            plt.plot(loss)
            plt.ylabel('training loss')
            plt.xlabel('epoch')
            plt.title('sigmoid net training loss')

            plt.show()
    elif args['model'] == 'baseline':

        baseline_err = fixed_baseline(test_data)
        print("  [Always NO baseline] test error is {} %".format(np.round(baseline_err * 100, 4)))
        lr_baseline_err = logistic_regression_baseline(train_data, test_data)
        print("  [Logistic Regression] test error is {} %".format(np.round(lr_baseline_err * 100, 4)))

    else:
        print("  [ERROR] model {} not supported...".format(args['model']))
