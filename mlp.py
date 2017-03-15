"""

3-layer MLP with two input units, ten hidden units, and one output unit.
Objective is to learn the concept of a circle in 2D space. A label of +1
is assigned if (x - a)^2 + (y - b)^2 < r^2 and is labeled -1 otherwise.

All data is drawn from the unit square, and a = 0.5, b = 0.6, and r = 0.4.

Training data is 100 random samples uniformly distributed on the unit square,
and test data is 100 random samples drawn similarly.
"""
from __future__ import division

import argparse
import matplotlib.pyplot as plt

from ops import *


def in_circle(x, y):
    d = np.power((x - 0.5), 2) + np.power((y - 0.6), 2)
    if d < np.power(0.4, 2):
        return 1.0
    else:
        return -1.0


def generate_data():
    train = np.random.uniform(0, 1, size=(100, 2))
    test = np.random.uniform(0, 1, size=(100, 2))
    train_labels = np.zeros((100, 1))
    test_labels = np.zeros((100, 1))
    for i in range(100):
        train_labels[i] = in_circle(train[i, 0], train[i, 1])
        test_labels[i] = in_circle(test[i, 0], test[i, 1])
    return train, train_labels, test, test_labels


def train_mlp(args):

    train_loss = np.zeros((int(args['n_epochs']), 1))
    test_loss = np.zeros_like(train_loss)
    err = np.zeros_like(train_loss)

    train_xs, train_ys, test_xs, test_ys = generate_data()

    # specify model
    w1 = random_normal(2, 10, stddev=2)
    w1_b = bias(10)
    w2 = random_normal(10, 1, stddev=2)
    w2_b = bias(1)

    grad_w1 = np.zeros_like(w1)
    grad_w1_b = np.zeros_like(w1_b)
    grad_w2 = np.zeros_like(w2)
    grad_w2_b = np.zeros_like(w2_b)

    for i in range(int(args['n_epochs'])):

        if i % 5000 == 0:
            print("  [*] Checkpoint: saving weights...")
            save_pkl(w1, 'models/mlp/w1-epoch-{}-seed-{}.pkl'.format(i, args['random_seed']))
            save_pkl(w2, 'models/mlp/w2-epoch-{}-seed-{}.pkl'.format(i, args['random_seed']))
            save_pkl(w1_b, 'models/mlp/w1_b-epoch-{}-seed-{}.pkl'.format(i, args['random_seed']))
            save_pkl(w2_b, 'models/mlp/w2_b-epoch-{}-seed-{}.pkl'.format(i, args['random_seed']))

        # zeros grads
        grad_w1[:] = 0.
        grad_w1_b[:] = 0.
        grad_w2[:] = 0.
        grad_w2_b[:] = 0.

        temp_train_loss = 0.

        for j in range(100):

            # x is a (1, 2) np.array
            x = train_xs[j]
            # forward equations
            h1 = sigmoid(np.add(np.matmul(w1.T, x), w1_b))
            h2 = sigmoid(np.add(np.matmul(w2.T, h1), w2_b))
            pred = binarize(h2, 0.5)
            temp_train_loss += mse(pred, train_ys[j])

            # compute gradients
            # grad_w2
            dE_dout = np.multiply(-1, pred - train_ys[j])
            dout_dnet2 = np.multiply(h2, 1. - h2)

            for k in range(10):
                dnet_dwk = h1[k]
                grad_w2[k] += np.multiply(dE_dout, np.multiply(dout_dnet2, dnet_dwk))

            # grad_w2_b
            grad_w2_b[0] += np.multiply(dE_dout, dout_dnet2)

            # grad_w1
            for k in range(10):
                # compute gradient of each w_kl in grad_w1 -> shape is (2, 10)
                dnet2_dh1 = w2[k]
                dh1_dnet1 = np.multiply(h1[k], 1 - h1[k])

                for l in range(2):
                    grad_w1[l, k] += np.multiply(dE_dout, np.multiply(dout_dnet2, np.multiply(dnet2_dh1,
                                                                      np.multiply(dh1_dnet1, x[l]))))

                grad_w1_b[k] += np.multiply(dE_dout, np.multiply(dout_dnet2, np.multiply(dnet2_dh1, dh1_dnet1)))

        # scale gradients and loss
        grad_w1 = np.divide(grad_w1, 100.)
        grad_w1_b = np.divide(grad_w1_b, 100.)
        grad_w2 = np.divide(grad_w2, 100.)
        grad_w2_b = np.divide(grad_w2_b, 100.)

        train_loss[i] += np.divide(temp_train_loss, 100.)

        # apply gradients
        lr = float(args['learning_rate'])
        w1 += np.multiply(lr, grad_w1)
        w1_b += np.multiply(lr, grad_w1_b)
        w2 += np.multiply(lr, grad_w2)
        w2_b += np.multiply(lr, grad_w2_b)

        # test
        wrong = 0.
        temp_test_loss = 0.

        for j in range(100):
            x = test_xs[j]
            # forward equations
            h1 = sigmoid(np.add(np.matmul(w1.T, x), w1_b))
            h2 = sigmoid(np.add(np.matmul(w2.T, h1), w2_b))
            pred = binarize(h2, 0.5)
            temp_test_loss += mse(pred, test_ys[j])

            if not pred == test_ys[j]:
                wrong += 1

        err[i] += np.round(np.divide(wrong, 100.) * 100, 4)
        test_loss[i] += np.divide(temp_test_loss, 100.)

        print("[Train] epoch: {}, loss: {}".format(i, train_loss[i]))
        print("[Test] epoch: {}, loss: {}, classification error: {} %".format(i, test_loss[i], err[i]))

        # re-shuffle data
        train = np.concatenate((train_xs, train_ys), axis=1)
        test = np.concatenate((test_xs, test_ys), axis=1)
        np.random.shuffle(train)
        np.random.shuffle(test)
        train_xs = train[:, 0:-1]
        train_ys = np.reshape(train[:, -1], (100, 1))
        test_xs = train[:, 0:-1]
        test_ys = np.reshape(train[:, -1], (100, 1))

    save_pkl(w1, 'models/mlp/w1-seed-{}.pkl'.format(args['random_seed']))
    save_pkl(w2, 'models/mlp/w2-seed-{}.pkl'.format(args['random_seed']))
    save_pkl(w1_b, 'models/mlp/w1_b-seed-{}.pkl'.format(args['random_seed']))
    save_pkl(w2_b, 'models/mlp/w2_b-seed-{}.pkl'.format(args['random_seed']))

    if args['plots']:
        plt.figure(1)
        plt.plot(err)
        plt.title('MLP classification error (5 random seeds ave)')
        plt.ylabel('percent')
        plt.xlabel('epoch')

        plt.figure(2)
        plt.subplot(121)
        plt.plot(train_loss)
        plt.ylabel('training loss')
        plt.xlabel('epoch')
        plt.title('MLP training loss (MSE) (5 random seeds ave)')

        plt.subplot(122)
        plt.plot(test_loss)
        plt.ylabel('test loss')
        plt.xlabel('epoch')
        plt.title('MLP test loss (MSE) (5 random seeds ave)')

        plt.show()


def forward(x1, x2):
    h1 = sigmoid(np.add(np.matmul(w1.T, np.array([x1, x2])), w1_b))
    h2 = sigmoid(np.add(np.matmul(w2.T, h1), w2_b))
    return binarize(h2, 0.5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set the run parameters.')
    parser.add_argument('--random_seed', default=1111)
    parser.add_argument('--n_epochs', default=20000)
    parser.add_argument('--plots', action='store_true', default=False)
    parser.add_argument('--learning_rate', default=0.025)
    parser.add_argument('--plot_weights', action='store_true')

    parser.set_defaults(plot_weights=False)
    parser.set_defaults(plots=True)

    args = vars(parser.parse_args())

    np.random.seed(int(args['n_epochs']))

    if not args['plot_weights']:
        train_mlp(args)
    else:
        w1 = load_pkl('models/mlp/w1-epoch-0-seed-82727.pkl')
        w2 = load_pkl('models/mlp/w2-epoch-0-seed-82727.pkl')
        w1_b = load_pkl('models/mlp/w1_b-epoch-0-seed-82727.pkl')
        w2_b = load_pkl('models/mlp/w2_b-epoch-0-seed-82727.pkl')

        circle = plt.Circle((0.5, 0.6), 0.4)
        fig, ax = plt.subplots()
        ax.add_artist(circle)

        for i in range(10):
            x1 = 0.
            x2 = 1.
            y1 = (-w1_b[i]-(w1[0][i]*x1))/w1[1][i]
            y2 = (-w1_b[i]-(w1[0][i]*x2))/w1[1][i]

            plt.plot([x1, x2], [y1, y2])

        plt.show()

