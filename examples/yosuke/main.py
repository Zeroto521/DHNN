# -*- coding: utf-8 -*-

from __future__ import print_function

import os

import dhnn

import utils

# paramters
cwd_path = os.path.split(os.path.abspath(__file__))[0]
threshold = 60
size = (100, 100)
pflag = 1
nflag = -1


if __name__ == "__main__":
    # create a list of train data.
    train_path = os.path.join(cwd_path, 'data/yosuke.jpg')
    train_data = utils.readImg2array(
        train_path, size, threshold, pflag=pflag, nflag=nflag)

    # create a list of tese data, e.g. yosukekatada 's sunglasses picture.
    test_path = os.path.join(cwd_path, 'data/sunglasses.jpg')
    test_data = utils.readImg2array(
        test_path, size, threshold, pflag=pflag, nflag=nflag)

    # build model
    model = dhnn.DHNN(pflag=pflag, nflag=nflag)

    print('[START] training.')
    model.train([train_data.flatten()])
    print('[END] training.')

    print('[START] predicting.')
    recovery = model.predict([test_data.flatten()], epochs=50000)
    print('[END] predicting.')

    print('[START] predicting.')
    recovery = recovery[0].reshape(size)
    outfile = os.path.join(cwd_path, 'data', 'recovery.jpg')
    utils.array2img(recovery, outFile=outfile, pflag=pflag, nflag=nflag)
    print('[END] saving.')
