#!/usr/bin/env python
"""
Implement a simple RNN to learn weights generated in the following manner

data = np.matrix(np.ones([200, 4]))

for i in range(0,200):
    data[i, 0] = random.random()
    data[i, 1] = random.random()

w1 = .23434
w1hat = -.3346
w2 = -.5757
w2hat = .5758
bias1 = .57686
bias2 = -.5767

for i in range(0, 200):

    Ofirst0 = 0
    Osecond0 = 0

    Ofirst1 = sigmoid(data[i, 0] * w1hat + bias1)
    Osecond1 = sigmoid(data[i, 1] * w2hat + bias2)

    Ofirst2 = sigmoid(w1 * Osecond1 + bias1)
    Osecond2 = sigmoid(w2 * Ofirst1 + bias2)

    Ofirst3 = sigmoid(bias1 + w1 * Osecond2)
    Osecond3 = sigmoid(bias2 + w2 * Ofirst2)

    data[i, 2] = Ofirst3
    data[i, 3] = Osecond3

"""
from ops import *
import pandas
import argparse
import matplotlib.pyplot as plt


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Set the run parameters.')
    parser.add_argument('--random_seed', default=12348)
    parser.add_argument('--dataset', default='data/q3data.csv', help='raw CSV file of data')
    parser.add_argument('--save_weights', action='store_true')
    parser.add_argument('--plots', action='store_true')
    parser.add_argument('--cv', default=10,
                        help='Number of times to do random train/test cross validation '
                             'splits before computing approximate model error')
    parser.add_argument('--n_epochs', default=1000)
    parser.add_argument('--beta', default=0.0001, help='L2 regularization parameter')
    parser.add_argument('--lr', default=0.001)

    parser.set_defaults(save_weights=False)
    parser.set_defaults(plots=False)

    args = vars(parser.parse_args())

    np.random.seed(int(args['random_seed']))

    data = np.matrix(pandas.read_csv((args['dataset'])))

    n_samples, _ = np.shape(data)

    # parameters
    w1 = random_normal(1, 1, stddev=1)
    w1_hat = random_normal(1, 1, stddev=1)
    w2 = random_normal(1, 1, stddev=1)
    w2_hat = random_normal(1, 1, stddev=1)
    bias1 = bias(1)
    bias2 = bias(1)

    # gradients
    grad_w1 = np.zeros_like(w1)
    grad_w1_hat = np.zeros_like(w1_hat)
    grad_w2 = np.zeros_like(w2)
    grad_w2_hat = np.zeros_like(w2_hat)
    grad_bias1 = np.zeros_like(bias1)
    grad_bias2 = np.zeros_like(bias2)

    grad_norms = np.zeros((6, int(args['n_epochs'])))
    w_norms = np.zeros_like(grad_norms)

    # do a random 70/30 split
    n_training_samples = int(0.7 * n_samples)
    n_test_samples = n_samples - n_training_samples
    print("Splitting dataset into train/test sets of sizes {}/{}".format(n_training_samples, n_test_samples))

    training_loss = np.zeros((2, int(args['n_epochs'])))
    test_loss = np.zeros_like(training_loss)

    for i in range(int(args['n_epochs'])):

        # zero grads
        grad_w1[:] = 0
        grad_w2[:] = 0
        grad_w1_hat[:] = 0
        grad_w2_hat[:] = 0
        grad_bias1[:] = 0
        grad_bias2[:] = 0

        # do args['cv'] splits, summing the loss
        for j in range(int(args['cv'])):

            np.random.shuffle(data)

            train_data = data[:n_training_samples]
            test_data = data[n_training_samples:]

            for k in range(n_training_samples):

                # forward equations
                Ofirst0 = 0
                Osecond0 = 0

                Ofirst1 = sigmoid(train_data[k, 0] * w1_hat + bias1)
                Osecond1 = sigmoid(train_data[k, 1] * w2_hat + bias2)

                Ofirst2 = sigmoid(w1 * Osecond1 + bias1)
                Osecond2 = sigmoid(w2 * Ofirst1 + bias2)

                Ofirst3 = sigmoid(w1 * Osecond2 + bias1)
                Osecond3 = sigmoid(w2 * Ofirst2 + bias2)

                training_loss[0][i] += mse(train_data[k, 2], Ofirst3)
                training_loss[1][i] += mse(train_data[k, 3], Osecond3)

                # accumulate gradients

                # grad_w1
                grad_w1_p1 = -(train_data[k, 3] - Osecond3) * Osecond3 * (1 - Osecond3) * w2 * Ofirst2 * (1 - Ofirst2) * Osecond1
                grad_w1_p2 = -(train_data[k, 2] - Ofirst3) * Ofirst3 * (1 - Ofirst3) * Osecond2
                grad_w1 += (grad_w1_p1 + grad_w1_p2)

                # grad_w2
                grad_w2_p1 = -(train_data[k, 3] - Osecond3) * Osecond3 * (1 - Osecond3) * Ofirst2
                grad_w2_p2 = -(train_data[k, 2] - Ofirst3) * Ofirst3 * (1 - Ofirst3) * w1 * Osecond2 * (1 - Osecond2) * Ofirst1
                grad_w2 += (grad_w2_p1 + grad_w2_p2)

                # grad_w1_hat
                grad_w1_hat += -(train_data[k, 2] - Ofirst3) * Ofirst3 * (1 - Ofirst3) * w1 * Osecond2 * \
                              (1 - Osecond2) * w2 * Ofirst1 * (1 - Ofirst1) * train_data[k, 0]

                # grad_w2_hat
                grad_w2_hat += -(train_data[k, 3] - Osecond3) * Osecond3 * (1 - Osecond3) * w2 * Ofirst2 * \
                              (1 - Ofirst2) * w1 * Osecond1 * (1 - Osecond1) * train_data[k, 1]

                # grad_bias1
                grad_bias1_p1 = -(train_data[k, 3] - Osecond3) * Osecond3 * (1 - Osecond3) * w2 * Ofirst2 * (1 - Ofirst2)
                grad_bias1_p2 = -(train_data[k, 2] - Ofirst3) * Ofirst3 * (1 - Ofirst3) * (1 + w1 * (Osecond2 *
                                                                             (1 - Osecond2) * Ofirst1 * (1 - Ofirst1)))
                grad_bias1 += np.reshape(grad_bias1_p1 + grad_bias1_p2, 1)

                # grad_bias2
                grad_bias2_p1 = -(train_data[k, 3] - Osecond3) * Osecond3 * (1 - Osecond3) * (1 + w2 * (Ofirst2 *
                                                                            (1 - Ofirst2) * Osecond1 * (1 - Osecond1)))
                grad_bias2_p2 = -(train_data[k, 2] - Ofirst3) * Ofirst3 * (1 - Ofirst3) * w1 * Osecond2 * (1 - Osecond2)
                grad_bias2 += np.reshape(grad_bias2_p1 + grad_bias2_p2, 1)

            for k in range(n_test_samples):
                # forward equations
                Ofirst0 = 0
                Osecond0 = 0

                Ofirst1 = sigmoid(test_data[k, 0] * w1_hat + bias1)
                Osecond1 = sigmoid(test_data[k, 1] * w2_hat + bias2)

                Ofirst2 = sigmoid(w1 * Osecond1 + bias1)
                Osecond2 = sigmoid(w2 * Ofirst1 + bias2)

                Ofirst3 = sigmoid(bias1 + w1 * Osecond2)
                Osecond3 = sigmoid(bias2 + w2 * Ofirst2)

                test_loss[0][i] += mse(test_data[k, 2], Ofirst3)
                test_loss[1][i] += mse(test_data[k, 3], Osecond3)

        training_loss[:, i] = np.divide(training_loss[:, i], n_training_samples * int(args['cv']))
        test_loss[:, i] = np.divide(test_loss[:, i], n_test_samples * int(args['cv']))

        train_l = np.round(training_loss[:, i], 4)
        test_l = np.round(test_loss[:, i], 4)

        print("\n")
        print("  epoch: {}, MSE training loss: {}, MSE test loss {}".format(i, train_l, test_l))

        # apply gradients

        # first, scale them
        grad_w1 = np.divide(grad_w1, n_training_samples * int(args['cv']))
        grad_w2 = np.divide(grad_w2, n_training_samples * int(args['cv']))
        grad_w1_hat = np.divide(grad_w1_hat, n_training_samples * int(args['cv']))
        grad_w2_hat = np.divide(grad_w2_hat, n_training_samples * int(args['cv']))
        grad_bias1 = np.divide(grad_bias1, n_training_samples * int(args['cv']))
        grad_bias2 = np.divide(grad_bias2, n_training_samples * int(args['cv']))

        grad_norms[0][i] = np.linalg.norm(grad_w1)
        grad_norms[1][i] = np.linalg.norm(grad_w2)
        grad_norms[2][i] = np.linalg.norm(grad_w1_hat)
        grad_norms[3][i] = np.linalg.norm(grad_w2_hat)
        grad_norms[4][i] = np.linalg.norm(grad_bias1)
        grad_norms[5][i] = np.linalg.norm(grad_bias2)

        w_norms[0][i] = np.linalg.norm(w1)
        w_norms[1][i] = np.linalg.norm(w2)
        w_norms[2][i] = np.linalg.norm(w1_hat)
        w_norms[3][i] = np.linalg.norm(w2_hat)
        w_norms[4][i] = np.linalg.norm(bias1)
        w_norms[5][i] = np.linalg.norm(bias2)

        print("[L2 norms] grad_w1: {}, grad_w2: {}, grad_w1_hat: {}, grad_w2_hat: {}, grad_bias1: {}, grad_bias2: {}"
              .format(grad_norms[0][i], grad_norms[1][i], grad_norms[2][i], grad_norms[3][i], grad_norms[4][i], grad_norms[5][i]))

        print("[L2 norms] w1: {}, w2: {}, w1_hat: {}, w2_hat: {}, bias1: {}, bias2: {}"
              .format(w_norms[0][i], w_norms[1][i], w_norms[2][i], w_norms[3][i], w_norms[4][i],
                      w_norms[5][i]))

        # gradient descent step
        w1 += float(args['lr']) * grad_w1 + 2 * float(args['beta']) * w1
        w2 += float(args['lr']) * grad_w2 + 2 * float(args['beta']) * w2
        w1_hat += float(args['lr']) * grad_w1_hat + 2 * float(args['beta']) * w1_hat
        w2_hat += float(args['lr']) * grad_w2_hat + 2 * float(args['beta']) * w2_hat
        bias1 += float(args['lr']) * grad_bias1 + 2 * float(args['beta']) * bias1
        bias2 += float(args['lr']) * grad_bias2 + 2 * float(args['beta']) * bias2

    if args['plots']:
        plt.figure(1)
        ax1 = plt.subplot(121)
        plt.plot(training_loss[0], label='y1')
        plt.plot(training_loss[1], label='y2')
        plt.ylabel('training loss')
        plt.xlabel('epoch')
        plt.title('RNN training loss')
        ax1.set_ylim([0, 0.01])

        ax2 = plt.subplot(122)
        plt.plot(test_loss[0], label='y1')
        plt.plot(test_loss[1], label='y2')
        plt.ylabel('test loss')
        plt.xlabel('epoch')
        plt.title('RNN test loss')
        ax2.set_ylim([0, 0.01])

        plt.show()
