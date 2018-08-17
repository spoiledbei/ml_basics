# https://deeplearningcourses.com/c/deep-learning-recurrent-neural-networks-in-python
# https://udemy.com/deep-learning-recurrent-neural-networks-in-python
import os

directory = '/Users/congshanzhang/Documents/Office/machine learning/dl_recurrent_nnet(code)'
os.chdir(directory)

from __future__ import print_function, division
from future.utils import iteritems
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


# =============================================================================
# word embedding:
#   1. word2idx is a dictionary {'START': 0, 'END': 1, 'king':2, 'man':3}
#   2. We is (V*D) matrix, thought of as a vocabulary matrix, which
#      maps each word in vocabulary into a D-dim vector
#   3. vec('king') = We[word2idx['king']]   
# =============================================================================

import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import init_weight, get_robert_frost, get_wikipedia_data


class SimpleRNN:
    def __init__(self, D, M, V):
        self.D = D # dimensionality of word embedding
        self.M = M # hidden layer size
        self.V = V # vocabulary size

    def fit(self, X, learning_rate=1., mu=0.99, reg=1.0, activation=T.tanh, epochs=500, show_fig=False):
        N = len(X) # total number of sentences; X is the entire document, with each row a sentence.
        D = self.D
        M = self.M
        V = self.V
        self.f = activation

        # initial weights
        We = init_weight(V, D)
        Wx = init_weight(D, M)
        Wh = init_weight(M, M)
        bh = np.zeros(M)
        h0 = np.zeros(M)
        Wo = init_weight(M, V)
        bo = np.zeros(V)

        # make them theano shared
        self.We = theano.shared(We)
        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.We, self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]

        # make inputs theano variables
        # thX: a sentence of document which consists of indices of words, 
        #    e.g. [0,5,4,7,11,3,4,1] 
        thX = T.ivector('X') # a sequence of indices of dimension T 
        Ei = self.We[thX]    # real input into the nnet; will be a TxD matrix
        thY = T.ivector('Y') # the target is the next word in the word list

        # sentence input:
        # [START, w1, w2, ..., wn]
        # sentence target:
        # [w1,    w2, w3, ..., END]

        def recurrence(x_t, h_t1):  # do calculation at each recurrent unit
            # returns h(t), y(t)
            h_t = self.f(x_t.dot(self.Wx) + h_t1.dot(self.Wh) + self.bh)  # here x_t is (1*D)
            y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
            return h_t, y_t

        [h, y], _ = theano.scan( # do calculation at each recurrent unit
            fn=recurrence,
            outputs_info=[self.h0, None],
            sequences=Ei,    # Ei is a (T*D) matrix. Each time, pass a row to recurrence function.
            n_steps=Ei.shape[0],
        )

        py_x = y[:, 0, :]  # py_x will be (T*V) 
        prediction = T.argmax(py_x, axis=1)  # prediction is (T*1)

        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY])) # cost for one sentence
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value()*0) for p in self.params]

        updates = []
        for p, dp, g in zip(self.params, dparams, grads):
            new_dp = mu*dp - lr*g
            updates.append((dp, new_dp))

            new_p = p + new_dp
            updates.append((p, new_p))

        # prediction is jsut one forward-feeding
        self.predict_op = theano.function(inputs=[thX], outputs=prediction)
        
        # training is forwardprop + backprop
        self.train_op = theano.function(
            inputs=[thX, thY],
            outputs=[cost, prediction],
            updates=updates
        )

        costs = [] # total cost of the document
        n_total = sum((len(sentence)+1) for sentence in X)  # +1 is to fill START/END
        for i in range(epochs):
            X = shuffle(X)  # shuffle the rows
            n_correct = 0
            cost = 0
            for j in range(N): # for each sentence
                # problem! many words --> END token are overrepresented
                # result: generated lines will be very short
                # we will try to fix in a later iteration
                # BAD! magic numbers 0 and 1...
                
                # we set 0 to start and 1 to end
                input_sequence = [0] + X[j]
                output_sequence = X[j] + [1]

                c, p = self.train_op(input_sequence, output_sequence)
                # print "p:", p
                cost += c
                # print "j:", j, "c:", c/len(X[j]+1)
                for pj, xj in zip(p, output_sequence):
                    if pj == xj:
                        n_correct += 1
            print("i:", i, "cost:", cost, "correct rate:", (float(n_correct)/n_total))
            costs.append(cost)

        if show_fig:
            plt.plot(costs)
            plt.show()
    
    def save(self, filename):
        np.savez(filename, *[p.get_value() for p in self.params]) # save multiple arrays at once

    @staticmethod
    def load(filename, activation):
        # TODO: would prefer to save activation to file too
        npz = np.load(filename)
        We = npz['arr_0']
        Wx = npz['arr_1']
        Wh = npz['arr_2']
        bh = npz['arr_3']
        h0 = npz['arr_4']
        Wo = npz['arr_5']
        bo = npz['arr_6']
        V, D = We.shape
        _, M = Wx.shape
        rnn = SimpleRNN(D, M, V)
        rnn.set(We, Wx, Wh, bh, h0, Wo, bo, activation)
        return rnn

    def set(self, We, Wx, Wh, bh, h0, Wo, bo, activation):
        self.f = activation

        # redundant - see how you can improve it
        self.We = theano.shared(We)
        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.We, self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]

        thX = T.ivector('X')
        Ei = self.We[thX] # will be a TxD matrix
        thY = T.ivector('Y')

        def recurrence(x_t, h_t1):
            # returns h(t), y(t)
            h_t = self.f(x_t.dot(self.Wx) + h_t1.dot(self.Wh) + self.bh)
            y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
            return h_t, y_t

        [h, y], _ = theano.scan(
            fn=recurrence,
            outputs_info=[self.h0, None],
            sequences=Ei,
            n_steps=Ei.shape[0],
        )

        py_x = y[:, 0, :]
        prediction = T.argmax(py_x, axis=1)
        
        self.predict_op = theano.function(
            inputs=[thX],
            outputs=prediction,
            allow_input_downcast=True,
        )

    def generate(self, pi, word2idx): # pi: the initial word distribution (prob of each word in vocab)
        # convert word2idx -> idx2word
        idx2word = {v:k for k,v in iteritems(word2idx)}
        V = len(pi)

        # generate 4 lines at a time
        n_lines = 0

        # why? because using the START symbol will always yield the same first word!
        # we choose a random word for each sentence.
        X = [ np.random.choice(V, p=pi) ] # choose a random word according the word distribution pi
        print(idx2word[X[0]], end=" ")

        while n_lines < 4:
            # print "X:", X is used to keep track of every line.
            P = self.predict_op(X)[-1]  # put the current word into the nnet and the prediction will be a scalar
            X += [P]
            if P > 1:
                # it's a real word, not start/end token
                word = idx2word[P]
                print(word, end=" ")
            elif P == 1:
                # end token
                n_lines += 1
                print('')
                if n_lines < 4:
                    X = [ np.random.choice(V, p=pi) ] # reset to start of line
                    print(idx2word[X[0]], end=" ")


def train_poetry():
    sentences, word2idx = get_robert_frost()
    rnn = SimpleRNN(30, 30, len(word2idx))
    rnn.fit(sentences, learning_rate=1e-4, show_fig=True, activation=T.nnet.relu, epochs=2000)
    rnn.save('RNN_D30_M30_epochs2000_relu.npz')

def generate_poetry():
    sentences, word2idx = get_robert_frost()
    rnn = SimpleRNN.load('RNN_D30_M30_epochs2000_relu.npz', T.nnet.relu)

    # determine initial state distribution for starting sentences, 
    # using each starting word for every sentence in the decument.
    V = len(word2idx)
    pi = np.zeros(V)
    for sentence in sentences:
        pi[sentence[0]] += 1
    pi /= pi.sum()

    rnn.generate(pi, word2idx)

def wikipedia():
    sentences, word2idx = get_wikipedia_data()
    print("finished retrieving data")
    print("vocab size:", len(word2idx), "number of sentences:", len(sentences))
    rnn = SimpleRNN(20, 15, len(word2idx))
    rnn.fit(sentences, learning_rate=1e-4, show_fig=True, activation=T.nnet.relu)

if __name__ == '__main__':
    train_poetry()
    generate_poetry()
    # wikipedia()
