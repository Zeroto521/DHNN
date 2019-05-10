# yosuke

An example that show how to strip off yosuke's sunglasses.

## Step1

Input a neat picture like this(yosukekatada's smile face).

![yosuke](data/yosuke.jpg)

## Step2

Get the network to memorize the pattern, this program will automatically transform RGB Jpg into black-white picture.

## Step3

After the network memorized it, put the picture with noise like this (yosukekatada's smile face with **sunglasses**) into the network.

![sunglasses](data/sunglasses.jpg)

## Step4

The network can strip off the sunglasses, because the network ready remembers the former picture.

![recovery](data/recovery.jpg)
