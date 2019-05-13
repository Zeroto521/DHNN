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
    >>> recovery = model.predict(test_data)  #  Guess you have `test_data` which the shape is `(m, 1)`.
    >>> recovery

Copyright Zeroto521
----------------------------
"""

import os

import numba as nb
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

        if isload and os.path.isfile(wpath):
            self._w = np.load(wpath)
        else:
            self._w = None

    @property
    def weight(self):
        return self._w

    @nb.autojit
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

    @nb.autojit
    def predict(self, data, theta=0.5, epochs=1000):
        """predict test sample.

        Arguments:
            data {np.ndarray} -- vector

        Keyword Arguments:
            theta {float} -- the threshold of the neuron activation(default: {0.5})
            epochs {int} -- the max iteration of loop(default: {1000})

        Returns:
            np.ndarray -- recoveried sample
        """

        # TODO: accelerate mulity-data predict by matrix dot method
        indexs = np.random.randint(0, len(data) - 1, epochs)
        for ind in indexs:
            vec = self._w[ind]
            value = vec.dot(data)
            data[ind] = self.pflag if (value - theta > 0) else self.nflag

        return data
