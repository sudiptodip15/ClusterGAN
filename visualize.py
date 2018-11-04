import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


def split(x):
    assert type(x) == int
    t = int(np.floor(np.sqrt(x)))
    for a in range(t, 0, -1):
        if x % a == 0:
            return a, x / a


def grid_transform(x, size):
    a, b = split(x.shape[0])
    h, w, c = size[0], size[1], size[2]
    x = np.reshape(x, [a, b, h, w, c])
    x = np.transpose(x, [0, 2, 1, 3, 4])
    x = np.reshape(x, [a * h, b * w, c])
    if x.shape[2] == 1:
        x = np.squeeze(x, axis=2)
    return x


def grid_show(fig, x, size):
    ax = fig.add_subplot(111)
    x = grid_transform(x, size)
    if len(x.shape) > 2:
        ax.imshow(x)
    else:
        ax.imshow(x, cmap='gray')



if __name__=='__main__':

    from keras.datasets import cifar10
    from scipy.misc import imsave
    import pdb

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    shape = x_train[0].shape
    bx = x_train[0:64,:]
    bx = grid_transform(bx, shape)

    imsave('cifar_batch.png', bx)

    pdb.set_trace()

    print('Done !')



