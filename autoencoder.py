import theano
import theano.tensor as T
import numpy as np
from util import *
from capsule import *
WEIGHT=0
BIAS  =1

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
        self.W0 = initialize_weight(n_visible  , n_hidden_l1, "W0", numpy_rng, 'uniform') 
        self.W1 = initialize_weight(n_hidden_l1, n_hidden_l2, "W1", numpy_rng, 'uniform')
        self.W2 = initialize_weight(n_hidden_l2, n_hidden_l3, "W2", numpy_rng, 'uniform')
        self.params0 = [self.W0, self.W1, self.W2,
                       self.l1_biases, self.l2_biases,self.l3_biases]

        #Adding Capsules
        self.capsules = []
        sphere1 = Capsule('sphere', n_hidden_l3, 6) #3 for center, 3 for scaling 
        self.capsules.append(sphere1)

        #self.l3_to_rvar2  = theano.shared(self.init_capsule_param(n_hidden_l3),borrow=True)
        #self.rvar2_biases = theano.shared(np.zeros(3), borrow=True)

        self.capsule_params = self._get_capsule_params()
        self.params= self.params0+self.capsule_params

    def _get_capsule_params(self):

        params = []
        for i in xrange(len(self.capsules)):
            params += self.capsules[i].params
        return params


    def get_reconstruct(self,X):
        robjs = self.encoder(X)
        return self.decoder(robjs)

    def encoder(self, X):

        h1 = T.tanh(T.dot(X , self.W0) + self.l1_biases)
        h2 = T.tanh(T.dot(h1, self.W1) + self.l2_biases)
        h3 = T.nnet.softplus(T.dot(h2, self.W2) + self.l3_biases)

        rvars = []
        #TODO For loop needs to be replaced with scan to make it faster
        for item_i in xrange(len(self.capsules)):
            obj_params = T.dot(h3, self.capsules[item_i].params[WEIGHT]) + self.capsules[item_i].params[BIAS]
            rvars.append(obj_params) 

        return rvars

    def decoder(self, robjs):
        return self.scene(self.capsules, robjs)

    def cost(self,  X):
        #robj1, robj2 = self.encoder(X)
        robjs = self.encoder(X)
        #reconImage = self.decoder(robj1, robj2)[:,:,0].flatten()
        reconImage = self.decoder(robjs)[:,:,0].flatten()
        return T.sum((X-reconImage)*(X-reconImage))


        #Should be this when we have multiple inputs NxD
        #return T.mean(0.5* T.sum((X-reconImage)*(X-reconImage),axis=1))
