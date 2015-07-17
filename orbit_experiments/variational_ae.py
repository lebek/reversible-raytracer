import theano
import theano.tensor as T
import numpy as np
from util import *
from capsule import *
import theano.sandbox.rng_mrg as RNG_MRG
numpy_rng = np.random.RandomState(1234)
MRG = RNG_MRG.MRG_RandomStreams(numpy_rng.randint(2 ** 30))

#CONSTANT
CWEIGHT=0
CBIAS  =1
RWEIGHT=2
RBIAS  =3

class VAE():

    # each render_var gets its own l2 layer
    def __init__(self, scene, n_visible,
                 n_hidden_l1, n_hidden_l2, n_hidden_l3, num_capsule):
        self.scene = scene
        self.n_visible = n_visible
        self.n_hidden_l1 = n_hidden_l1
        self.n_hidden_l2 = n_hidden_l2
        self.n_hidden_l3 = n_hidden_l3

        self.l1_biases  = theano.shared(np.zeros(n_hidden_l1, dtype=theano.config.floatX), borrow=True)
        self.l2_biases  = theano.shared(np.zeros(n_hidden_l2, dtype=theano.config.floatX), borrow=True)
        self.l3_biases  = theano.shared(np.zeros(n_hidden_l3, dtype=theano.config.floatX), borrow=True)
        self.mu_biases  = theano.shared(np.zeros(n_hidden_l3, dtype=theano.config.floatX), borrow=True)
        self.var_biases = theano.shared(np.zeros(n_hidden_l3, dtype=theano.config.floatX), borrow=True)

        self.W0 = initialize_weight(n_visible  , n_hidden_l1, "W0", numpy_rng, 'uniform') 
        self.W1 = initialize_weight(n_hidden_l1, n_hidden_l2, "W1", numpy_rng, 'uniform')
        self.W2 = initialize_weight(n_hidden_l2, n_hidden_l3, "W1", numpy_rng, 'uniform')
        self.W_mu  = initialize_weight(n_hidden_l3, n_hidden_l3, "W_mu", numpy_rng, 'uniform')
        self.W_var = initialize_weight(n_hidden_l3, n_hidden_l3, "W_var", numpy_rng, 'uniform')
        self.params0 = [self.W0, self.W1, self.W_mu, self.W_var,
                       self.l1_biases, self.l2_biases,self.mu_biases, self.var_biases] \
                        + [self.W2, self.l3_biases]

        #Adding Capsules
        self.capsules = []
        for i in xrange(num_capsule):
            sphere = Capsule('sphere', n_hidden_l3, 6, num_capsule) #3 for center, 3 for scaling 
            self.capsules.append(sphere)

        self.capsule_params = self._get_capsule_params()
        self.params= self.params0+self.capsule_params

    def _get_capsule_params(self):

        params = []
        for i in xrange(len(self.capsules)):
            params += self.capsules[i].params
        return params

    def KL_Q_P(self,H):
        var  = T.exp(0.5*T.dot(H,self.W_var)+ self.var_biases)
        mean = T.dot(H,self.W_mu) + self.mu_biases
        return 0.5*T.sum(mean**2 + var - T.log(var)-1, axis=1)

    def get_reconstruct(self,X):
        h3 = self.encoder_pre(X.dimshuffle('x',0))
        robjs = self.encoder_post(h3)
        return self.decoder(robjs)

    def encoder(self,X):
        return self.encoder_post(self.encoder_pre(X))

    def encoder_pre(self, X):

        h1 = T.nnet.softplus(T.dot(X, self.W0) + self.l1_biases)
        h2 = T.nnet.softplus(T.dot(h1, self.W1) + self.l2_biases)
        h3 = T.nnet.softplus(T.dot(h2, self.W2) + self.l3_biases)
        return h3

    def encoder_post(self, h):
        z = self.get_latent(h)
        rvars = []
        #TODO For loop needs to be replaced with scan to make it faster
        for item_i in xrange(len(self.capsules)):
            center = T.dot(z, self.capsules[item_i].params[CWEIGHT]) \
                                    + self.capsules[item_i].params[CBIAS]
            center = T.set_subtensor(center[:,2], T.nnet.softplus(center[:,2]))
            rvars.append(center.flatten()) 

        return rvars

    def get_latent(self, h):

        sigs = MRG.normal(size=(1, self.n_hidden_l3), avg=0., std=1.)
        log_sigma_encoder = 0.5*(T.dot(h, self.W_var) + self.var_biases)
        mu = T.dot(h, self.W_mu) + self.mu_biases

        #Find the hidden variable z
        z = mu + T.exp(log_sigma_encoder)*sigs
        return z 

    def decoder(self, robjs):
        return self.scene(self.capsules, robjs)

    def cost(self,  X):
        h3 = self.encoder_pre(X.dimshuffle('x',0))
        robjs = self.encoder_post(h3)
        reconImage = self.decoder(robjs).flatten()

        lossZ= T.mean(T.sum(self.KL_Q_P(h3),axis=0))
        lossX= T.mean(-T.sum(X * T.log(reconImage) + (1-X)*T.log(1-reconImage), axis=0))

        return lossX + lossZ


        #Should be this when we have multiple inputs NxD
        #return T.mean(0.5* T.sum((X-reconImage)*(X-reconImage),axis=1))
