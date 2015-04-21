import numpy as np
import theano
import theano.tensor as T
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def transNorm(transM, vec):


    transN = T.zeros_like(vec)
    transN = T.set_subtensor(transN[:,:,0], vec[:,:,0] * transM[0][0] \
                                + vec[:,:,1] * transM[1][0] + vec[:,:,2] * transM[2][0])
    transN = T.set_subtensor(transN[:,:,1], vec[:,:,0] * transM[0][1] \
                                + vec[:,:,1] * transM[1][1] + vec[:,:,2] * transM[2][1])
    transN = T.set_subtensor(transN[:,:,2], vec[:,:,0] * transM[0][2] \
                                + vec[:,:,1] * transM[1][2] + vec[:,:,2] * transM[2][2])

    return transN

<<<<<<< HEAD
def draw(fname, im):

=======
def drawWithMarkers(fname, im):
>>>>>>> 3b6a34095f5f207dd306d7217c04b558e384cd52
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im, interpolation='nearest')
    ax.add_patch(plt.Rectangle((85-3, 90-3), 6, 6, color='red',
                               linewidth=2, fill=False))
    ax.add_patch(plt.Rectangle((90-3, 50-3), 6, 6, color='red',
                               linewidth=2, fill=False))
    fig.savefig(fname, bbox_inches='tight', pad_inches=0)

def draw(fname, im):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im, interpolation='nearest')
    fig.savefig(fname, bbox_inches='tight', pad_inches=0)
