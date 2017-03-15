#!/usr/bin/env python
from __future__ import division
from __future__ import absolute_import

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


def pla(args, random_seeds, train_data, test_data):
    """
    Perceptron Learning Algorithm 

    :param args:
    :param train_data:
    :param test_data:
    """
    err = np.zeros((int(args['n_epochs']), 1))
    train_loss = np.zeros_like(err)
    test_loss = np.zeros_like(err)

    for seed in random_seeds:

        np.random.seed(seed)

        data = prep_data(train_data, test_data)
        n_features = data['train_cols'] - 2

        params = np.zeros((n_features, 1), dtype=np.float32)

        for i in range(int(args['n_epochs'])):

            temp_train_loss = 0.

            for j in range(data['train_rows']):
                out = sign(np.dot(data['train_xs'][j, :], params))
                temp_train_loss += mse(out, data['train_ys'][j])

                if not data['train_ys'][j] == out:
                    # params update is: params <= params + y * x
                    params = np.add(params, np.reshape(np.multiply(data['train_ys'][j],
                                                                   data['train_xs'][j, :]), (n_features, 1)))

            # re-normalize the parameter vector so ||theta|| = 1
            params = np.divide(params, np.linalg.norm(params))
            train_loss[i] += np.divide(temp_train_loss, data['train_rows'])

            wrong = 0.
            temp_test_loss = 0.
            for j in range(data['test_rows']):
                out = sign(np.dot(data['test_xs'][j, :], params))
                temp_test_loss += mse(out, data['test_ys'][j])

                if not data['test_ys'][j] == out:
                    wrong += 1

            test_loss[i] += np.divide(temp_test_loss, data['test_rows'])
            err[i] += np.round(np.divide(wrong, data['test_rows']) * 100, 4)

            # print("  [PLA] validation error of {} % at epoch {}".format(err[i], i))

    err = np.divide(err, len(random_seeds))
    train_loss = np.divide(train_loss, len(random_seeds))
    test_loss = np.divide(test_loss, len(random_seeds))

    return err, train_loss, test_loss


def sigmoid_net(args, random_seeds, train_data, test_data):
    """
    Use the sigmoid activation function instead of 
    sign function

    :param args:
    :param train_data:
    :param test_data:
    """
    err = np.zeros(int(args['n_epochs']))
    train_loss = np.zeros_like(err)
    test_loss = np.zeros_like(err)

    for seed in random_seeds:

        np.random.seed(seed)

        data = prep_data(train_data, test_data, augment=True)

        n_features = data['train_cols'] - 1
        # Add an extra parameter for the bias terms
        # params = np.zeros((cols-1, 1), dtype=np.float32)
        params = np.random.normal(scale=1, size=(n_features, 1))
        grads = np.zeros_like(params)

        for i in range(int(args['n_epochs'])):

            grads[:] = 0.
            temp_train_loss = 0.

            for j in range(data['train_rows']):
                out = sigmoid(np.dot(data['train_xs'][j, :], params))
                pred = binarize(out, 0.5)
                temp_train_loss += mse(pred, data['train_ys'][j])

                # gradient computation
                for k in range(n_features):
                    dE_dout = np.multiply(-1, pred - data['train_ys'][j])
                    dout_dnet = np.multiply(out, 1. - out)
                    dnet_dwk = data['train_xs'][j, k]
                    grads[k] += np.multiply(dE_dout, np.multiply(dout_dnet, dnet_dwk))

            scaled_grads = np.divide(grads, data['train_rows'])
            train_loss[i] += np.divide(temp_train_loss, data['train_rows'])

            # print("  [Sigmoid] training loss {} at epoch {}".format(np.round(loss[i], 4), i))

            # apply gradients
            lr = float(args['learning_rate'])
            params += np.multiply(lr, scaled_grads)

            # compute validation mis-classification error
            wrong = 0.
            temp_test_loss = 0.
            for j in range(data['test_rows']):
                out = sigmoid(np.dot(data['test_xs'][j, :], params))
                pred = binarize(out, 0.5)
                temp_test_loss += mse(pred, data['test_ys'][j])
                if not data['test_ys'][j] == pred:
                    wrong += 1

            err[i] += np.round(np.divide(wrong, data['test_rows']) * 100, 4)
            test_loss[i] += np.divide(temp_test_loss, data['test_rows'])

            # print("  [Sigmoid] test error of {} % at epoch {}".format(err[i], i))

            # Re-shuffle the data for the next epoch
            data = prep_data(train_data, test_data, augment=True)

    err = np.divide(err, len(random_seeds))
    train_loss = np.divide(train_loss, len(random_seeds))
    test_loss = np.divide(test_loss, len(random_seeds))
    return err, train_loss, test_loss


def fixed_baseline(random_seeds, train_data, test_data):
    """Guess no every time."""

    err = 0.

    for seed in random_seeds:
        np.random.seed(seed)

        data = prep_data(train_data, test_data)

        wrong = 0.
        for i in range(data['test_rows']):
            if not data['test_ys'][i] == 1.0:
                wrong += 1

        err += wrong / data['test_rows']

    return err / len(random_seeds)


def logistic_regression_baseline(random_seeds, train_data, test_data):
    # use default parameters
    lr = LogisticRegression()

    err = 0.

    for seed in random_seeds:
        np.random.seed(seed)

        data = prep_data(train_data, test_data)

        lr.fit(data['train_xs'], data['train_ys'])
        accuracy = lr.score(data['test_xs'], data['test_ys'])

    err += 1. - accuracy

    return err / len(random_seeds)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Set the run parameters.')
    parser.add_argument('--model', default='PLA')
    parser.add_argument('--dataset', default='data/bank.csv', help='raw CSV file of data')
    parser.add_argument('--training_data', default='data/bank_train.pkl',
                        help='specify pickled preprocessed training data')
    parser.add_argument('--test_data', default='data/bank_test.pkl', help='specify pickled preprocessed test data')
    parser.add_argument('--n_epochs', default=500, help='num epochs of training')
    parser.add_argument('--learning_rate', default=0.1, help='learning rate for gradient descent')
    parser.add_argument('--plots', action='store_true', default=False)

    parser.set_defaults(plots=True)

    args = vars(parser.parse_args())

    random_seeds = [1234, 1337, 4911, 9991, 2940]

    if os.path.isfile(os.path.join(os.getcwd(), args['training_data'])) and \
            os.path.isfile(os.path.join(os.getcwd(), args['test_data'])):
        train_data = load_pkl(args['training_data'])
        test_data = load_pkl(args['test_data'])
    else:
        train_data, test_data = preprocess(args['dataset'])

    if args['model'] == 'PLA':
        err, train_loss, test_loss = pla(args, random_seeds, train_data, test_data)

        print("[PLA] test classification error: {}, train_loss: {}, test_loss: {}".format(err[-1], train_loss[-1], test_loss[-1]))
        if args['plots']:
            plt.figure(1)
            plt.plot(err)
            plt.title('PLA test classification error (5 random seeds ave)')
            plt.ylabel('percent')
            plt.xlabel('epoch')

            plt.figure(2)
            plt.subplot(121)
            plt.plot(train_loss)
            plt.ylabel('training loss')
            plt.xlabel('epoch')
            plt.title('PLA training loss (MSE) (5 random seeds ave)')

            plt.subplot(122)
            plt.plot(test_loss)
            plt.ylabel('test loss')
            plt.xlabel('epoch')
            plt.title('PLA test loss (MSE) (5 random seeds ave)')

            plt.show()

    elif args['model'] == 'sigmoid':
        err, train_loss, test_loss = sigmoid_net(args, random_seeds, train_data, test_data)
        print("[sigmoid] test classification error: {}, train_loss: {}, test_loss: {}".format(err[-1], train_loss[-1], test_loss[-1]))
        if args['plots']:
            plt.figure(1)
            plt.plot(err)
            plt.title('Sigmoid test classification error (5 random seeds ave) lr: {}'.format(args['learning_rate']))
            plt.ylabel('percent')
            plt.xlabel('epoch')

            plt.figure(2)
            plt.subplot(121)
            plt.plot(train_loss)
            plt.ylabel('training loss')
            plt.xlabel('epoch')
            plt.title('Sigmoid training loss (MSE) (5 random seeds ave) lr: {}'.format(args['learning_rate']))

            plt.subplot(122)
            plt.plot(test_loss)
            plt.ylabel('test loss')
            plt.xlabel('epoch')
            plt.title('Sigmoid test loss (MSE) (5 random seeds ave) lr: {}'.format(args['learning_rate']))

            plt.show()

    elif args['model'] == 'baselines':

        baseline_err = fixed_baseline(random_seeds, train_data, test_data)
        print("  [Always NO baseline] classification error is {} %".format(baseline_err * 100))
        lr_baseline_err = logistic_regression_baseline(random_seeds, train_data, test_data)
        print("  [Logistic Regression] test error is {} %".format(lr_baseline_err * 100))

    else:
        print("  [ERROR] model {} not supported...".format(args['model']))
