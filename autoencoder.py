import theano
import theano.tensor as T
import numpy as np
from util import *

class Autoencoder():

    # each render_var gets its own l2 layer
    def __init__(self, scene, n_visible,
                 n_hidden_l1, n_hidden_l2, n_hidden_l3):
        self.scene = scene
        self.n_visible = n_visible
        self.n_hidden_l1 = n_hidden_l1
        self.n_hidden_l2 = n_hidden_l2

        self.l1_biases = theano.shared(np.zeros(n_hidden_l1), borrow=True)
        self.l2_biases = theano.shared(np.zeros(n_hidden_l2), borrow=True)
        self.l3_biases = theano.shared(np.zeros(n_hidden_l3), borrow=True)

        numpy_rng = np.random.RandomState(1234)
        self.vis_to_l1 = initialize_weight(n_visible, n_hidden_l1, "vis_to_l1", numpy_rng, 'uniform')
        self.l1_to_l2 = initialize_weight(n_hidden_l1, n_hidden_l2, "vis_to_l1", numpy_rng, 'uniform')
        self.l2_to_l3 = initialize_weight(n_hidden_l2, n_hidden_l3, "vis_to_l1", numpy_rng, 'uniform')

        self.params0 = [self.vis_to_l1, self.l1_to_l2, self.l2_to_l3,
                       self.l1_biases, self.l2_biases,self.l3_biases]


        #Adding Capsules
        self.l3_to_rvar1  = theano.shared(self.init_capsule_param(n_hidden_l3),borrow=True)
        self.rvar1_biases = theano.shared(np.asarray([0,0,2.5,1,1,1]), borrow=True)

        #self.l3_to_rvar2  = theano.shared(self.init_capsule_param(n_hidden_l3),borrow=True)
        #self.rvar2_biases = theano.shared(np.zeros(3), borrow=True)

        self.params1 = [self.l3_to_rvar1,  self.rvar1_biases]
                        #self.l3_to_rvar2, self.rvar2_biases]
        self.params= self.params0+self.params1

    def init_capsule_param(self, n_hidden_l3):
        l3_to_center = 0.07*np.asarray(
            np.random.uniform(
                low=-4 * np.sqrt(6. / 6+n_hidden_l3),
                high=4 * np.sqrt(6. / 6+n_hidden_l3),
                size=(n_hidden_l3, 3)
            ), dtype=theano.config.floatX)
        l3_to_radius = 0.0007*np.asarray(
            np.random.uniform(
                low=-4 * np.sqrt(6. / 6+n_hidden_l3),
                high=4 * np.sqrt(6. / 6+n_hidden_l3),
                size=(n_hidden_l3, 3)
            ), dtype=theano.config.floatX)
        return np.concatenate((l3_to_center, l3_to_radius), 1)

    def get_reconstruct(self,X):
        robj1 = self.encoder(X)
        return self.decoder(robj1)

    def encoder(self, X):

        h1 = T.nnet.sigmoid(T.dot(X, self.vis_to_l1) + self.l1_biases)
        h2 = T.nnet.sigmoid(T.dot(h1, self.l1_to_l2) + self.l2_biases)
        h3 = T.nnet.sigmoid(T.dot(h2, self.l2_to_l3) + self.l3_biases)
        rvar1 = T.dot(h3, self.l3_to_rvar1) + self.rvar1_biases
        #rvar2 = T.dot(h3, self.l3_to_rvar2) + self.rvar2_biases

        #Assume all objects are within 20m from the camara
        #rvar1 = T.set_subtensor(rvar1[2], rvar1[2].clip(2.1,5))
        #rvar2 = T.set_subtensor(rvar2[2], rvar2[2].clip(2.1,5))
        return rvar1#,rvar2

    def decoder(self, robj1):
        return self.scene(robj1[:3], robj1[3:])

    def cost(self,  X):
        #robj1, robj2 = self.encoder(X)
        robj1 = self.encoder(X)
        #reconImage = self.decoder(robj1, robj2)[:,:,0].flatten()
        reconImage = self.decoder(robj1)[:,:,0].flatten()
        return T.sum((X-reconImage)*(X-reconImage))


        #Should be this when we have multiple inputs NxD
        #return T.mean(0.5* T.sum((X-reconImage)*(X-reconImage),axis=1))
