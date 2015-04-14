import numpy as np
import theano
import theano.tensor as T

def transNorm(transM, vec):


    transN = T.zeros_like(vec)
    transN = T.set_subtensor(transN[:,:,0], vec[:,:,0] * transM[0][0] \
                                + vec[:,:,1] * transM[1][0] + vec[:,:,2] * transM[2][0])
    transN = T.set_subtensor(transN[:,:,1], vec[:,:,0] * transM[0][1] \
                                + vec[:,:,1] * transM[1][1] + vec[:,:,2] * transM[2][1])
    transN = T.set_subtensor(transN[:,:,2], vec[:,:,0] * transM[0][2] \
                                + vec[:,:,1] * transM[1][2] + vec[:,:,2] * transM[2][2])	

    return transN


