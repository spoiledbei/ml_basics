# https://deeplearningcourses.com/c/deep-learning-recurrent-neural-networks-in-python
# https://udemy.com/deep-learning-recurrent-neural-networks-in-python
import os

directory = '/Users/congshanzhang/Documents/Office/machine learning/dl_recurrent_nnet(code)'
os.chdir(directory)

from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import init_weight, all_parity_pairs_with_sequence_labels


class SimpleRNN:
    def __init__(self, M):
        self.M = M    # hidden layer size within each recurrent unit

    def fit(self, X, Y, learning_rate=0.1, mu=0.99, reg=1.0, activation=T.tanh, epochs=100, show_fig=False):
        D = X[0].shape[1]           # X is of size N x T(n) x D
        K = len(set(Y.flatten()))   # K is the total number of classes
        N = len(Y)                  # N is the total number of individuals
        M = self.M
        self.f = activation

        # initial weights (3 Weights, 2 biases, 1 hidden state)
        Wx = init_weight(D, M)
        Wh = init_weight(M, M)
        bh = np.zeros(M)
        h0 = np.zeros(M)            # initial hidden state
        Wo = init_weight(M, K)
        bo = np.zeros(K)

        # make them theano shared (3 Weights, 2 biases, 1 hidden state)
        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]

        thX = T.fmatrix('X')     # 2-dimensional (T*D). In parity problem, D=1, e.g. thX = [[0],[1],[1],[0]]
        thY = T.ivector('Y')     # thY in parity problem is also a sequence thY = [0,1,0,0]

        def recurrence(x_t, h_t1): # do calculation at each recurrent unit
            # returns h(t), y(t)
            h_t = self.f(x_t.dot(self.Wx) + h_t1.dot(self.Wh) + self.bh)
            y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
            return h_t, y_t

        [h, y], _ = theano.scan(  # do calculation at each recurrent unit
            fn=recurrence,
            outputs_info=[self.h0, None],
            sequences=thX,        # e.g. [[0],[1],[1],[0]]
            n_steps=thX.shape[0], # for each time
        )

        py_x = y[:, 0, :]
        prediction = T.argmax(py_x, axis=1)

        # cost for one sequence
        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))  # cross-entropy loss
        grads = T.grad(cost, self.params)  # gradient
        dparams = [theano.shared(p.get_value()*0) for p in self.params] # momentum

        updates = [
            (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
        ]

        self.predict_op = theano.function(inputs=[thX], outputs=prediction) # forward feeding one time
        self.train_op = theano.function(
            inputs=[thX, thY],
            outputs=[cost, prediction, y],  # forwardprop to get loss
            updates=updates                 # backprop to update params
        )

        # total cost for all sequences
        costs = []
        # main training loop
        for i in range(epochs):
            X, Y = shuffle(X, Y)
            n_correct = 0
            cost = 0
            for j in range(N): # train one sequence at a time! So we do not specify batch_size, since batch_size = 1.
                c, p, rout = self.train_op(X[j], Y[j]) # each time we train a sequence, we update all params.
                # print "p:", p
                cost += c
                if p[-1] == Y[j,-1]: # only need to check the last element in p and Y for parity problem
                    n_correct += 1
            print("shape y:", rout.shape)
            print("i:", i, "cost:", cost, "accuracy:", (float(n_correct)/N))
            costs.append(cost)
            if n_correct == N:
                break

        if show_fig:
            plt.plot(costs)
            plt.show()


def parity(B=12, learning_rate=1e-4, epochs=200):  # B: number of bits
    X, Y = all_parity_pairs_with_sequence_labels(B)

    rnn = SimpleRNN(20)  # size of hidden layer = 20
    rnn.fit(X, Y, learning_rate=learning_rate, epochs=epochs, activation=T.nnet.relu, show_fig=True)


if __name__ == '__main__':
    parity()
    
    

