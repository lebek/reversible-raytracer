import theano
import theano.tensor as T
import numpy as np
from util import *
from capsule import *

#CONSTANT
CWEIGHT=0
CBIAS  =1
RWEIGHT=2
RBIAS  =3

class LinEncoder():

    # each render_var gets its own l2 layer
    def __init__(self, scene, n_visible,
                 n_hidden_l1, num_capsule):
        self.scene = scene
        self.n_visible = n_visible
        self.n_hidden_l1 = n_hidden_l1

        self.l1_biases = theano.shared(np.zeros(n_hidden_l1, dtype=theano.config.floatX), borrow=True)

        numpy_rng = np.random.RandomState(123)
        self.W0 = initialize_weight(n_visible  , n_hidden_l1, "W0", numpy_rng, 'uniform') 
        self.params0 = [self.W0, 
                       self.l1_biases]

        #Adding Capsules
        self.capsules = []
        for i in xrange(num_capsule):
            sphere = Capsule('sphere', n_hidden_l1, 3, num_capsule) #3 for center, 3 for scaling 
            self.capsules.append(sphere)

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
        robjs = self.encoder(X.dimshuffle('x',0))
        return self.decoder(robjs)

    def encoder(self, X):

        h1 = T.tanh(T.dot(X , self.W0) + self.l1_biases)

        rvars = []
        #TODO For loop needs to be replaced with scan to make it faster
        for item_i in xrange(len(self.capsules)):
            center = T.dot(h1, self.capsules[item_i].params[CWEIGHT]) \
                                    + self.capsules[item_i].cbias
            #center = T.set_subtensor(center[:,2], T.nnet.softplus(center[:,2]))
            center = T.set_subtensor(center[:,2], T.switch(center[:,2]<0, 0, center[:,2]))
            rvars.append(center.flatten()) 

        return rvars

    def decoder(self, robjs):
        return self.scene(self.capsules, robjs)

    def cost(self,  X):
        robjs = self.encoder(X.dimshuffle('x',0))
        reconImage = self.decoder(robjs).flatten()
        return T.sum((X-reconImage)*(X-reconImage)) #- 0.00001 * T.sum((robjs[0][0] + robjs[0][1])**2)


        #Should be this when we have multiple inputs NxD
        #return T.mean(0.5* T.sum((X-reconImage)*(X-reconImage),axis=1))
