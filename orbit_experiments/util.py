import numpy as np
import theano
import theano.tensor as T
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.misc import imsave


def initialize_weight(n_vis, n_hid, W_name, numpy_rng, rng_dist):

    if 'uniform' in rng_dist:
        W = numpy_rng.uniform(low=-np.sqrt(6. / (n_vis + n_hid)),\
                high=np.sqrt(6. / (n_vis + n_hid)),
                size=(n_vis, n_hid)).astype(theano.config.floatX)
    elif rng_dist == 'normal':
        W = 0.01 * numpy_rng.normal(size=(n_vis, n_hid)).astype(theano.config.floatX)

    return theano.shared(value = W, name=W_name, borrow=True)


'''decaying learning rate'''
def get_epsilon(epsilon, n, i):
    return float(epsilon / ( 1 + i/float(n)))


def broadcasted_switch(a, b, c):
    return T.switch(a.dimshuffle(0, 1, 'x'), b, c)


def transNorm(transM, vec):


    transN = T.zeros_like(vec)
    transN = T.set_subtensor(transN[:,:,0], vec[:,:,0] * transM[0][0] \
                                + vec[:,:,1] * transM[1][0] + vec[:,:,2] * transM[2][0])
    transN = T.set_subtensor(transN[:,:,1], vec[:,:,0] * transM[0][1] \
                                + vec[:,:,1] * transM[1][1] + vec[:,:,2] * transM[2][1])
    transN = T.set_subtensor(transN[:,:,2], vec[:,:,0] * transM[0][2] \
                                + vec[:,:,1] * transM[1][2] + vec[:,:,2] * transM[2][2])

    return transN

def drawWithMarkers(fname, im):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im, interpolation='nearest')
    ax.add_patch(plt.Rectangle((85-3, 90-3), 6, 6, color='red',
                               linewidth=2, fill=False))
    ax.add_patch(plt.Rectangle((90-3, 50-3), 6, 6, color='red',
                               linewidth=2, fill=False))
    fig.savefig(fname, bbox_inches='tight', pad_inches=0)

def draw(fname, im):
    imsave(fname, im)

