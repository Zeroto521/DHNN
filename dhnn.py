# -*- coding: utf-8 -*-

"""
DHNN
=====
A Discrete Hopfield Neural Network Framework in python.

Example
----------------------------
    >>> from dhnn import DHNN
    >>> model = dhnn.DHNN()  # build model
    >>> model.train(train_data)  # Guess you have `train_data` which the shape is `(n, m)`. `n` is sample numbers, `m` is feature numbers and each sample must be vector.
    >>> recovery = model.predict(test_data)  #  Guess you have `test_data` which the shape is `(n, m)`. `n` and `m` is same means to train_data's shape.
    >>> recovery

Copyright Zeroto521
----------------------------
"""


import numpy as np

__version__ = '0.1.11'
__license__ = 'MIT'
__short_description__ = 'A Discrete Hopfield Neural Network Framework in python.'


class DHNN(object):

    def __init__(self, isload=False, wpath='weigh.npy', pflag=1, nflag=0):
        """Initializes DHNN.

        Keyword Arguments:
            isload {bool} -- is load local weight (default: {False})
            wpath {str} -- the local weight path (default: {'weigh.npy'})
            pflag {int} -- positive flag (default: 1)
            nflag {int} -- negative flag (default: -1)
        """

        self.pflag = pflag
        self.nflag = nflag

        if isload:
            from os.path import isfile
            if isfile(wpath):
                self._w = np.load(wpath)
        else:
            self._w = None

    @property
    def weight(self):
        return self._w

    def train(self, data, issave=False, wpath='weigh.npy'):
        """Training pipeline.

        Arguments:
            data {list} -- each sample is vector

        Keyword Arguments:
            issave {bool} -- save weight or not (default: {True})
            wpath {str} -- the local weight path (default: {'weigh.npy'})
        """

        mat = np.vstack(data)
        eye = len(data) * np.identity(np.size(mat, 1))
        self._w = np.dot(mat.T, mat) - eye

        if issave:
            np.save(wpath, self.weight)

    def predict(self, data, theta=0.5, epochs=1000):
        """predict sample.

        Arguments:
            data {np.ndarray} -- vector

        Keyword Arguments:
            theta {float} -- the threshold of the neuron activation(default: {0.5})
            epochs {int} -- the max iteration of loop(default: {1000})

        Returns:
            np.ndarray -- recoveried sample
        """

        if isinstance(data, list):
            data = np.asarray(data)
            print(data.shape)

        indexs = np.random.randint(0, len(self._w) - 1, (epochs, len(data)))
        for ind in indexs:
            diagonal = np.diagonal(np.dot(self._w[ind], data.T))
            diagonal = np.expand_dims(diagonal, -1)
            value = np.apply_along_axis(
                lambda x: self.pflag if x > theta else self.nflag, 1, diagonal)

            for i in range(len(data)):
                data[i, ind[i]] = value[i]

        return data
