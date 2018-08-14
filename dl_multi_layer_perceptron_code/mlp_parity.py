import os

directory = '/Users/congshanzhang/Documents/Office/machine learning/dl_multi_layer_perceptron(lowerlevelcode)'
os.chdir(directory)

# https://deeplearningcourses.com/c/deep-learning-recurrent-neural-networks-in-python
# https://udemy.com/deep-learning-recurrent-neural-networks-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from util import init_weight, all_parity_pairs
from sklearn.utils import shuffle


class HiddenLayer(object):
    def __init__(self, M1, M2, an_id): 
        self.id = an_id
        self.M1 = M1  # input size
        self.M2 = M2  # output size
        W = init_weight(M1, M2)   # initialize a weight matrix
        b = np.zeros(M2)     # initialize the bias vector
        self.W = theano.shared(W, 'W_%s' % self.id)
        self.b = theano.shared(b, 'b_%s' % self.id)
        self.params = [self.W, self.b]

    def forward(self, X):  # X is an np.array (N*M1)
        return T.nnet.relu(X.dot(self.W) + self.b)


class ANN(object): # Artificial neural network
    def __init__(self, hidden_layer_sizes): # hidden_layer_sizes is a list of numbers specifying # of nodes for each layer (not including the last classification layer)
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self, X, Y, learning_rate=1e-2, mu=0.99, reg=1e-12, epochs=400, batch_sz=20, print_period=1, show_fig=False):

        # X = X.astype(np.float32)
        Y = Y.astype(np.int32)

        # initialize hidden layers
        N, D = X.shape
        K = len(set(Y))
        self.hidden_layers = []  # list of hidden layers
        M1 = D
        count = 0
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2, count)  # use 'count' as layer_id
            self.hidden_layers.append(h)    # append newly defined layer h into the list
            M1 = M2   # update input size for the next layer
            count += 1
        W = init_weight(M1, K)   # This is for the last classification layer 
        b = np.zeros(K)
        self.W = theano.shared(W, 'W_logreg')
        self.b = theano.shared(b, 'b_logreg')

        # collect params for later use
        self.params = [self.W, self.b]
        for h in self.hidden_layers:    # for each layer
            self.params += h.params     # add into the parameter list the Weight and Bias of each layer

        # for momentum
        dparams = [theano.shared(np.zeros(p.get_value().shape)) for p in self.params]

        # for rmsprop
        cache = [theano.shared(np.zeros(p.get_value().shape)) for p in self.params]

        # set up theano functions and variables
        thX = T.matrix('X')
        thY = T.ivector('Y')   # the labels are presented as 1D vector of [int] labels
        pY = self.forward(thX)

        # regularization cost
        rcost = reg*T.sum([(p*p).sum() for p in self.params])  # L2-loss: square all parameters in the NN and sum
        # total cost = cross entropy + regularization cost
        cost = -T.mean(T.log(pY[T.arange(thY.shape[0]), thY])) + rcost
        # note: T.arange(thY.shape[0]) is a 1D vector [0,1,...,N]; thY is a 1D vector of integer labels [2,1,1,3,1,2,2,3,...] of size N.
        prediction = self.predict(thX)
        # calculate gradient: Theano.grad cost wrt params
        grads = T.grad(cost, self.params)

        # momentum only
        """ note: the usage of zip()
            >>>a = [1,2,3]
            >>> b = [4,5,6]
            >>> zipped = zip(a,b)
                [(1, 4), (2, 5), (3, 6)]
            """
        updates = [
                   (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)  # update params
        ] + [
            (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
        ] # update momentum pramas

        train_op = theano.function(
            inputs=[thX, thY],
            outputs=[cost, prediction],  # forwardprop to get output
            updates=updates,             # backprop to update params
        )

        n_batches = N // batch_sz
        costs = []
        # training loop
        for i in range(epochs): # for each epoch
            X, Y = shuffle(X, Y)
            for j in range(n_batches): # for each batch
                Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
                Ybatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]

                c, p = train_op(Xbatch, Ybatch)

                if j % print_period == 0:  # if print_period = 10, then print every 10 batches
                    costs.append(c)
                    e = np.mean(Ybatch != p)
                    print("i:", i, "j:", j, "nb:", n_batches, "cost:", c, "error rate:", e)
        
        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        Z = X
        for h in self.hidden_layers: # for each layer:
            Z = h.forward(Z)         #      calculate relu(X.dot(W)+b)
        return T.nnet.softmax(Z.dot(self.W) + self.b)  # return the output of the last classification layer

    def predict(self, X):
        pY = self.forward(X)         # get predicted probabilities 
        return T.argmax(pY, axis=1)  # get predicted class 


def wide(): # one inner layer model
    X, Y = all_parity_pairs(12)
    model = ANN([2048])
    model.fit(X, Y, learning_rate=1e-4, print_period=10, epochs=300, show_fig=True)

def deep():
    # Challenge - find a deeper, slimmer network to solve the problem
    X, Y = all_parity_pairs(12)
    model = ANN([1024]*2)
    model.fit(X, Y, learning_rate=1e-3, print_period=10, epochs=100, show_fig=True)

if __name__ == '__main__':
    wide()
    # deep()
