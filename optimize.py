import os
import numpy as np
import theano.tensor as T
import theano
from autoencoder import Autoencoder

from transform import *
from scene import *
from shader import *

class GDOptimizer:
    """
    Gradient Descent optimizer
    """

    def __init__(self):
        pass

    def optimize(self, tVars, loss, lr, momentum=0):
        """
        Maximize the loss function with learning rate lr.
        """
        updateVars = []
        grads = T.grad(loss, tVars)
        for var, gvar in zip(tVars, grads):
            updateVars.append((var, var - lr * gvar))

        return theano.function([], loss, updates=updateVars)


    def ADAMopt(self, tVars, loss, lr, momentum=0):

        i = T.iscalar('i'); lr = T.fscalar('lr');
        grads = T.grad(loss, tVars)
        '''ADAM Code from 
            https://github.com/danfischetti/deep-recurrent-attentive-writer/blob/master/DRAW/adam.py
        '''
        self.m = [theano.shared(name = 'm', \
                value = np.zeros(param.get_value().shape,dtype=theano.config.floatX)) for param in model.params]
        self.v = [theano.shared(name = 'v', \
        	value = np.zeros(param.get_value().shape,dtype=theano.config.floatX)) for param in model.params]
        self.t = theano.shared(name = 't',value = np.asarray(1).astype(theano.config.floatX))
        updates = [(self.t,self.t+1)] 

        for param, gparam,m,v in zip(model.params, gparams, self.m, self.v):

            b1_t = 1-(1-beta1)*(l**(self.t-1))
            m_t = b1_t*gparam + (1-b1_t)*m
            updates.append((m,m_t))
            v_t = beta2*(gparam**2)+(1-beta2)*v
            updates.append((v,v_t))
            m_t_bias = m_t/(1-(1-beta1)**self.t)	
            v_t_bias = v_t/(1-(1-beta2)**self.t)
            if param.get_value().ndim == 1:
                updates.append((param,param - 5*lr*m_t_bias/(T.sqrt(v_t_bias)+epsilon)))		
            else:
                updates.append((param,param - lr*m_t_bias/(T.sqrt(v_t_bias)+epsilon)))		

        return theano.function([], loss, updates=updates)


    def ADAM(self, model, train_set, valid_set, test_set, \
                            beta1 = 0.1,beta2 = 0.001,epsilon = 1e-8, l = 1e-8):
 
        i = T.iscalar('i'); lr = T.fscalar('lr');
        X = T.matrix('X'); Y = T.matrix('X')

        cost, reconX, C, updates_scan = model.cost(X, Y)
        cost_test, _, _, _ = model.cost(X, Y)
        gparams = T.grad(cost, model.params)

        '''ADAM Code from 
            https://github.com/danfischetti/deep-recurrent-attentive-writer/blob/master/DRAW/adam.py
        '''
        self.m = [theano.shared(name = 'm', \
                value = np.zeros(param.get_value().shape,dtype=theano.config.floatX)) for param in model.params]
        self.v = [theano.shared(name = 'v', \
        	value = np.zeros(param.get_value().shape,dtype=theano.config.floatX)) for param in model.params]
        self.t = theano.shared(name = 't',value = np.asarray(1).astype(theano.config.floatX))
        updates = [(self.t,self.t+1)] 

        for param, gparam,m,v in zip(model.params, gparams, self.m, self.v):

            b1_t = 1-(1-beta1)*(l**(self.t-1))
            m_t = b1_t*gparam + (1-b1_t)*m
            updates.append((m,m_t))
            v_t = beta2*(gparam**2)+(1-beta2)*v
            updates.append((v,v_t))
            m_t_bias = m_t/(1-(1-beta1)**self.t)	
            v_t_bias = v_t/(1-(1-beta2)**self.t)
            if param.get_value().ndim == 1:
                updates.append((param,param - 5*lr*m_t_bias/(T.sqrt(v_t_bias)+epsilon)))		
            else:
                updates.append((param,param - lr*m_t_bias/(T.sqrt(v_t_bias)+epsilon)))		

        train_update = theano.function([i, theano.Param(lr,default=self.epsilon)],\
                outputs=cost, updates=updates,\
                givens={ X:train_set[0][i*self.batch_sz:(i+1)*self.batch_sz], \
                         Y:train_set[1][i*self.batch_sz:(i+1)*self.batch_sz], })

        get_valid_cost   = theano.function([i], outputs=cost_test,\
                givens={ X:valid_set[0][i*self.batch_sz:(i+1)*self.batch_sz], \
                         Y:valid_set[1][i*self.batch_sz:(i+1)*self.batch_sz], })

        updates.append(updates_scan)
        get_reconX  = theano.function([], outputs=reconX, givens={ X:valid_set[0][:self.batch_sz], Y:valid_set[1][:self.batch_sz] })
        get_canvases= theano.function([], outputs=C     , givens={ X:valid_set[0][:self.batch_sz], Y:valid_set[1][:self.batch_sz] })
        return train_update, get_valid_cost, get_reconX, get_canvases




class MGDAutoOptimizer:

    def __init__(self, ae):
        self.ae = ae

    def optimize(self, train_data, lr):

        X = T.fvector('X')
        #i = T.lscalar('i')
        cost = self.ae.cost(X)
        grads = T.grad(cost, self.ae.params)
        update_vars = []
        for var, gvar in zip(self.ae.params, grads):
            update_vars.append((var, var-lr*gvar))

        #optimize = theano.function([i], cost, updates=updateVars,
        #            given={X:train_data[i*batch_sz:(i+1)*batch_sz]}

        opt = theano.function([], cost, updates=update_vars,
                              givens={X: train_data[0]})

        get_grad = theano.function([], grads[3], givens={X:train_data[0]})
        get_gradb = theano.function([], grads[-1], givens={X:train_data[0]})
        return opt, get_grad, get_gradb



from scipy import misc
train_data = [misc.imread('15.jpg').flatten().astype('float32')/255.0]
#train_data = [misc.imread('15.png').flatten().astype('float32')/255.0]

#from scipy import ndimage
#train_data = [ndimage.imread('output/0.jpg', mode='RGB')[:,:,0].flatten().astype('float32')/255.0]
#import pdb; pdb.set_trace()
#center1 = theano.shared(np.asarray([-.5, -.5, 4], dtype=theano.config.floatX),
#                       borrow=True)

def scene(center1, center2):

    material1 = Material((0.2, 0.9, 0.4), 0.3, 0.7, 0.5, 50.)

    t1 = translate(center1)
    t2 = translate(center2)

    shapes = [
        Sphere(t1, material1), Sphere(t2, material1)
    ]

    light = Light((-1., -1., 2.), (0.961, 1., 0.87))
    camera = Camera(32, 32)
    shader = PhongShader()
    scene = Scene(shapes, [light], camera, shader)
    return scene.build()

if not os.path.exists('output'):
    os.makedirs('output')

ae = Autoencoder(scene, 32*32, 300, 30, 10)
opt = MGDAutoOptimizer(ae)
train_ae, get_grad, get_gradb = opt.optimize(train_data, 0.01)

get_recon = theano.function([], ae.get_reconstruct(train_data[0]))
get_centre1 = theano.function([], ae.encoder(train_data[0])[0])
get_centre2 = theano.function([], ae.encoder(train_data[0])[1])
get_cost  = theano.function([], ae.cost(train_data[0]))

n=0;
#center_i1 =get_centre1()
#center_i2 =get_centre2()
#print '...Epoch %d Train loss %g, Center1 (%g, %g, %g), Center1 (%g, %g, %g)' \
#                    % (n, get_cost(),center_i1[0], center_i1[1], center_i1[2],\
#                                     center_i2[0], center_i2[1], center_i2[2])

while (n<3):
    n+=1
    ggg =get_grad()
    gbb =get_gradb()
    import pdb; pdb.set_trace()

    train_loss = train_ae()
    #center_i1 =get_centre1()
    #center_i2 =get_centre2()
    #print '...Epoch %d Train loss %g, Center1 (%g, %g, %g), Center1 (%g, %g, %g)' \
    #                % (n, get_cost(),center_i1[0], center_i1[1], center_i1[2],\
    #                                 center_i2[0], center_i2[1], center_i2[2])

    image = get_recon()
    imsave('output/test%d.png' % (n,), image)
