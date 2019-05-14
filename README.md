# DHNN

A **D**iscrete **H**opfield **N**eural **N**etwork Framework in python.

[![Build Status](https://travis-ci.com/Zeroto521/DHNN.svg?branch=master)](https://travis-ci.com/Zeroto521/DHNN) [![](https://img.shields.io/pypi/v/dhnn.svg)](https://pypi.org/project/dhnn/)

## Introduction

DHNN is a minimalistic and Numpy based implementation of the [Discrete Hopfield Network](http://en.wikipedia.org/wiki/Hopfield_network). DHNN can learn (memorize) patterns and remember (recover) the patterns when the network feeds those with noises.

## Installation

Just use pip:

```bash
pip install dhnn
```

Or download `dhnn` to a directory which your choice and use `setup` to install script:

```bash
>>> git clone https://github.com/Zeroto521/DHNN.git
>>> python setup.py install
```

## Prerequisites

Prior to running this package, please install the following libraries.

-   `numpy`
-   `numba`

## Authors

| <img src="https://avatars3.githubusercontent.com/u/4463558?v=4" alt="yosukekatada" width="100px" height="100px"/> | <img src="https://avatars1.githubusercontent.com/u/25895405?v=4" alt="Zeroto521" width="100px" height="100px"/> |
| :---------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------: |
|                                  [yosukekatada](https://github.com/yosukekatada)                                  |                                    [Zeroto521](https://github.com/Zeroto521)                                    |

## TODO

-   [x] more flag, add 0/1 flag or other flag.
-   [ ] optimize loop, try numba, Cpython or any other ways.
-   [ ] optimize memory.

## License

MIT License. [@yosukekatada](https://github.com/yosukekatada), [@Zeroto521](https://github.com/Zeroto521)

## References

-   http://rishida.hatenablog.com/entry/2014/03/03/174331
