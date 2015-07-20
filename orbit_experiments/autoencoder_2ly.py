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

class Autoencoder2ly():

    # each render_var gets its own l2 layer
    def __init__(self, scene, n_visible,
                 n_hidden_l1, n_hidden_l2, num_capsule):
        self.scene = scene
        self.n_visible = n_visible
        self.n_hidden_l1 = n_hidden_l1
        self.n_hidden_l2 = n_hidden_l2

        self.l1_biases = theano.shared(np.zeros(n_hidden_l1, dtype=theano.config.floatX), borrow=True)
        self.l2_biases = theano.shared(np.zeros(n_hidden_l2, dtype=theano.config.floatX), borrow=True)

        numpy_rng = np.random.RandomState(123)
        self.W0 = initialize_weight(n_visible  , n_hidden_l1, "W0", numpy_rng, 'uniform') 
        self.W1 = initialize_weight(n_hidden_l1, n_hidden_l2, "W1", numpy_rng, 'uniform')

        self.params0 = [self.W0, self.W1, self.l1_biases]#, self.l2_biases]

        #Adding Capsules
        self.capsules = []
        for i in xrange(num_capsule):
            sphere = Capsule('sphere', n_hidden_l2, 6, num_capsule) #3 for center, 3 for scaling 
            self.capsules.append(sphere)

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
        h2 = T.tanh(T.dot(h1, self.W1) + self.l2_biases)
        #h2 = T.switch(h2<0, 0., h2)

        rvars = []
        #TODO For loop needs to be replaced with scan to make it faster
        for item_i in xrange(len(self.capsules)):
            center = T.dot(h2, self.capsules[item_i].params[CWEIGHT]) \
                                    + self.capsules[item_i].cbias
            #center = T.set_subtensor(center[:,2], T.nnet.softplus(center[:,2]))
            center = T.set_subtensor(center[:,2], T.switch(center[:,2]<0, 0, center[:,2]))
            rvars.append(center.flatten()) 
            #scale  = T.dot(h3, self.capsules[item_i].params[RWEIGHT]) \
            #                                + self.capsules[item_i].params[RBIAS]
            #rvars.append(T.stacklists([center, scale])) 

        return rvars

    def decoder(self, robjs):
        return self.scene(self.capsules, robjs)

    def cost(self,  X):
        robjs = self.encoder(X.dimshuffle('x',0))
        reconImage = self.decoder(robjs).flatten()
        return T.sum((X-reconImage)*(X-reconImage)) #- 0.00001 * T.sum((robjs[0][0] + robjs[0][1])**2)


        #Should be this when we have multiple inputs NxD
        #return T.mean(0.5* T.sum((X-reconImage)*(X-reconImage),axis=1))



    def penality(self):

        penTerms = []
        objects = self.scene_obj.shapes
        for i in xrange(len(objects)-1):

            obj1 = objects[i] 
            for j in xrange(i,len(objects)):

                obj2 = objects[j]
               
                #Distance check between objects
                center1 = obj1.o2w.m[:,3]
                center2 = obj2.o2w.m[:,3]
                radius1 = T.maximum(obj1.o2w.m[0,0], obj1.o2w.m[1,1])
                radius2 = T.maximum(obj2.o2w.m[0,0], obj2.o2w.m[1,1])
            
                max_rad = T.maximum(radius1, radius2)
                
                #TODO remake it for batch
                dist = T.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                penflag = T.switch(dist < max_rad, 1, 0)
        
                penTerms = T.sum(penflag * (np.pi * max_rad**2)*2)

        return penTerms 


    def out_of_boundary_penality(self):

        penTerms = T.constant(0)
        objects = self.scene_obj.shapes

        for i in xrange(len(objects)):

            obj1    = objects[i] 
            rf      = obj1.w2o(self.scene_obj.camera.rays)
            ttt     = obj1._hit(rf.rays,rf.origin) 
            penflag = T.switch(T.sum(ttt) <= 0, 1, 0)
            penTerms = penTerms + penflag * 100 
          
        return penTerms
                












