# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image


def readImg2array(file, size, threshold=145, pflag=1, nflag=-1):
    """Read Image file and convert it to Numpy array.

    Arguments:
        file {str} -- image local file path
        size {tuple} -- resize image size

    Keyword Arguments:
        threshold {int} -- binary threshold (default: {145})

    Returns:
        np.ndarray -- image matrix
    """

    # read image file and convert it to gray
    img = Image.open(file).convert(mode="L")
    img = img.resize(size)  # resize image

    # binarize image array
    imgArray = np.asarray(img)
    mat = np.zeros(img.size)

    mat[imgArray > threshold] = pflag
    mat[imgArray <= threshold] = nflag

    return mat


def array2img(data, outFile=None, pflag=1, nflag=-1):
    """Convert Numpy array to Image file like `*.jpg`.

    Arguments:
        data {np.ndarray} -- 1 or -1 matrix

    Keyword Arguments:
        outFile {str} -- output file path (default: {None})

    Returns:
        Image.image -- imgage
    """

    mat = np.zeros(data.shape, dtype=np.uint8)
    mat[data == pflag] = 255
    mat[data == nflag] = 0
    img = Image.fromarray(mat)
    if outFile is not None:
        img.save(outFile)

    return img
