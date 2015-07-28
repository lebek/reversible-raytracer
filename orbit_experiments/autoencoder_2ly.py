import theano
import theano.tensor as T
import numpy as np
from util import *
from capsule import *
from transform import *

#CONSTANT
CWEIGHT=0
CBIAS  =1
RWEIGHT=2
RBIAS  =3
CAM1LOC = translate((0, 2.5,0))
CAM2LOC = translate((0,-2.5,0))
CAM1DIR = np.asarray([0,0,1], dtype='float32')
CAM2DIR = np.asarray([0,0,1], dtype='float32')


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


    def get_reconstruct(self,Xl, Xr):
        robjsl, robjsr = self.encoder(Xl.dimshuffle('x',0), Xr.dimshuffle('x',0))
        reconImage_l, reconImage_r = self.decoder(robjsl, robjsr)
        reconImage_l = reconImage_l[1].flatten()
        reconImage_r = reconImage_r[1].flatten()
        return reconImage_l, reconImage_r

    def encoder(self, Xl, Xr):

        h1l = T.tanh(T.dot(Xl, self.W0) + self.l1_biases)
        h2l = T.tanh(T.dot(h1l, self.W1) + self.l2_biases)

        h1r = T.tanh(T.dot(Xr, self.W0) + self.l1_biases)
        h2r = T.tanh(T.dot(h1r, self.W1) + self.l2_biases)

        rvars_l = []; rvars_r = []
        #TODO For loop needs to be replaced with scan to make it faster
        for item_i in xrange(len(self.capsules)):
            center_l = T.dot(h2l, self.capsules[item_i].params[CWEIGHT]) + self.capsules[item_i].cbias
            center_l = T.set_subtensor(center_l[:,2], T.switch(center_l[:,2]<0, 0, center_l[:,2]))
            center_r = T.dot(h2r, self.capsules[item_i].params[CWEIGHT]) + self.capsules[item_i].cbias
            center_r = T.set_subtensor(center_r[:,2], T.switch(center_r[:,2]<0, 0, center_r[:,2]))
            rvars_l.append(center_l.flatten()) 
            rvars_r.append(center_r.flatten()) 

        return rvars_l, rvars_r

    def decoder(self, robjs_l, robjs_r):
        return self.scene(self.capsules, robjs_l, CAM1LOC, CAM1DIR), \
                self.scene(self.capsules, robjs_r, CAM2LOC, CAM2DIR)

    def cost(self,  Xl, Xr):
        robjsl, robjsr = self.encoder(Xl.dimshuffle('x',0), Xr.dimshuffle('x',0))
        reconImage_l, reconImage_r = self.decoder(robjsl, robjsr)
        reconImage_l = reconImage_l[1].flatten()
        reconImage_r = reconImage_r[1].flatten()
        return T.sum((Xl-reconImage_l)*(Xl-reconImage_l)) + T.sum((Xr-reconImage_r)*(Xr-reconImage_r))

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
                












