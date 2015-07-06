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
        cost = self.ae.cost(X)
        grads = T.grad(cost, self.ae.params)
        update_vars = []

        for var, gvar in zip(self.ae.params, grads):
            update_vars.append((var, var-lr*gvar))

        opt = theano.function([], cost, updates=update_vars, givens={X: train_data[0]})

        get_grad = theano.function([] , grads[-2], givens={X:train_data[0]})
        get_gradb = theano.function([], grads[-1], givens={X:train_data[0]})
        return opt, get_grad, get_gradb

    def optimizeADAM(self, train_data, lr, \
            beta1 = 0.1,beta2 = 0.001,epsilon = 1e-8, l = 1e-8):

        X = T.fvector('X')
        #i = T.lscalar('i')
        cost = self.ae.cost(X)
        grads = T.grad(cost, self.ae.params)

        '''ADAM Code from 
            https://github.com/danfischetti/deep-recurrent-attentive-writer/blob/master/DRAW/adam.py
        '''
        self.m = [theano.shared(name = 'm', \
                value = np.zeros(param.get_value().shape,dtype=theano.config.floatX)) for param in self.ae.params]
        self.v = [theano.shared(name = 'v', \
        	value = np.zeros(param.get_value().shape,dtype=theano.config.floatX)) for param in self.ae.params]
        self.t = theano.shared(name = 't',value = np.asarray(1).astype(theano.config.floatX))
        updates = [(self.t,self.t+1)] 

        for param, gparam,m,v in zip(self.ae.params, grads, self.m, self.v):

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


        opt = theano.function([], cost, updates=updates,
                              givens={X: train_data[0]})

        get_grad = theano.function([], grads[-2], givens={X:train_data[0]})
        get_gradb = theano.function([], grads[-1], givens={X:train_data[0]})
        return opt, get_grad, get_gradb





#train_data = [misc.imread('15.png').flatten().astype('float32')/255.0]


