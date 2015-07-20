import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
import numpy as np
from util import *
from capsule import *

#CONSTANT
CWEIGHT=0
CBIAS  =1
RWEIGHT=2
RBIAS  =3

class Conv_encoder():

    # each render_var gets its own l2 layer
    def __init__(self, scene, n_visible, num_capsule, num_channel=3,\
                        filter_shape1=(2,3,5,5), filter_shape2=(1,2,3,3), batch_sz=1, pool_size=(2,2)):
        self.scene = scene
        self.n_visible = n_visible

        D = int(np.sqrt(n_visible / num_channel))
        self.image_shape1   = (batch_sz, 3, D, D)
        self.filter_shape1  = filter_shape1
        f0 = self.filter_shape1[-1]
        output_size1   = (D - f0 + 1) / pool_size[0]


        self.image_shape2   = (batch_sz, 2, output_size1, output_size1)
        self.filter_shape2  = filter_shape2
        self.pool_size      = pool_size  
        f1 = self.filter_shape2[-1]
        output_size2   = (output_size1 - f1 + 1) / pool_size[0]
        num_hid = output_size2 * output_size2
        print 'Layer0: Filter size %d, image size out %d' % (f0, output_size1)
        print 'Layer1: Filter size %d, image size out %d' % (f1, output_size2)

        numpy_rng = np.random.RandomState(123)
        self.W0 = initalize_conv_weight(filter_shape1, pool_size, numpy_rng)
        self.W1 = initalize_conv_weight(filter_shape2, pool_size, numpy_rng)
        #self.W2 = initialize_weight(num_hid, num_hid, 'W2', numpy_rng, 'uniform')
        self.params0 = [self.W0, self.W1]#, self.l1_biases]#, self.l2_biases]

        #Adding Capsules
        self.capsules = []
        for i in xrange(num_capsule):
            sphere = Capsule('sphere', num_hid, 6, num_capsule) #3 for center, 3 for scaling 
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

        X  = X.reshape(self.image_shape1)
        h1 = self.convProp(X , self.W0, self.image_shape1, self.filter_shape1, self.pool_size)
        h2 = self.convProp(h1, self.W1, self.image_shape2, self.filter_shape2, self.pool_size).flatten(2)
        #h3 = T.tanh(T.dot(h2, self.W2))

        rvars = []
        #TODO For loop needs to be replaced with scan to make it faster
        for item_i in xrange(len(self.capsules)):
            center = T.dot(h2, self.capsules[item_i].params[CWEIGHT]) \
                                    + self.capsules[item_i].cbias
            #center = T.set_subtensor(center[:,2], T.nnet.softplus(center[:,2]))
            center = T.set_subtensor(center[:,2], T.switch(center[:,2]<0, 0, center[:,2]))
            rvars.append(center.flatten()) 

        return rvars

    def decoder(self, robjs):
        self.scene_obj, image = self.scene(self.capsules, robjs) 
        return image

    def cost(self,  X):
        robjs = self.encoder(X.dimshuffle('x',0))
        reconImage = self.decoder(robjs).flatten()
        return T.sum((X-reconImage)*(X-reconImage)) 
    
        #Should be this when we have multiple inputs NxD
        #return T.mean(0.5* T.sum((X-reconImage)*(X-reconImage),axis=1))


    def penalty(self):

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
        
                #Computing the overlapping area of two circle (is not working...)
                #R = radius1 
                #r = radius2
                #A = r**2 * T.arccos( (dist**2 + r**2 - R**2)/(2*dist*r)) \
                #        + R**2 * T.arccos( (dist**2-R**2 +r**2) / (2*dist*R) )\
                #        - 0.5 * T.sqrt((-dist + r +R)*(dist+r+R)*(dist-r+R)*(dist+r+R))
        
                penTerms = T.sum(penflag * (np.pi * max_rad**2)*2)# + (1-penflag) * A )

        return penTerms 


    def out_of_boundary_penalty(self):

        penTerms = T.constant(0)
        objects = self.scene_obj.shapes

        for i in xrange(len(objects)):

            obj1    = objects[i] 
            rf      = obj1.w2o(self.scene_obj.camera.rays)
            ttt     = obj1._hit(rf.rays,rf.origin) 
            penflag = T.switch(T.sum(ttt) <= 0, 1, 0)
            penTerms = penTerms + penflag * 100 
          
        return penTerms
                


    def convProp(self, X, W, image_shape, filter_shape, pool_size):

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=X,
            filters=W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=pool_size,
            ignore_border=True
        )

        return T.tanh(pooled_out)


