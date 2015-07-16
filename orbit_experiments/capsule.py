import numpy as np
import theano
import theano.tensor as T

class Capsule():

    def __init__(self, name, n_hidden, n_output, num_caps):

        self.name = name
        #bias = np.asarray([-2,2, 10.5 * num_caps,3.5,3.5,3.5], dtype=theano.config.floatX)/ num_caps
        cbias = np.asarray([0, 0, 40], dtype=theano.config.floatX)/ num_caps
        #rbias = np.asarray([ 2, 2, 2], dtype=theano.config.floatX)/ num_caps
        self.params = [self.init_capsule_cweight(n_hidden), theano.shared(cbias, borrow=True, name='cbias')]#,\
                       #self.init_capsule_rweight(n_hidden), theano.shared(rbias, borrow=True, name='rbias')]


    def init_capsule_cweight(self, n_hidden_l3):    

        l3_to_center = 0.1*np.asarray(
            np.random.uniform(
                low=-4 * np.sqrt(6. / 3+n_hidden_l3),
                high=4 * np.sqrt(6. / 3+n_hidden_l3),
                size=(n_hidden_l3, 3)
            ), dtype=theano.config.floatX)

        return theano.shared(l3_to_center, name='Cweight')

    def init_capsule_rweight(self, n_hidden_l3):
        l3_to_radius = 0.002*np.asarray( np.random.uniform(
                low=-4 * np.sqrt(6. / 3+n_hidden_l3),
                high=4 * np.sqrt(6. / 3+n_hidden_l3),
                size=(n_hidden_l3, 3)
            ), dtype=theano.config.floatX)
        return theano.shared(l3_to_radius, name='Rweight')


        #return theano.shared(0.07*np.asarray(
        #        np.random.uniform(
        #            low=-4 * np.sqrt(6. / n_output+n_hidden_l3),
        #            high=4 * np.sqrt(6. / n_output+n_hidden_l3),
        #            size=(n_hidden_l3, n_output)
        #        ), dtype=theano.config.floatX),borrow=True)



