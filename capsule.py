import numpy as np
import theano
import theano.tensor as T

class Capsule():

    def __init__(self, name, n_hidden, n_output):

        self.name = name
        self.params = [self.init_capsule_weight(n_hidden, n_output),
            theano.shared(np.asarray([0,0,2.5,1,1,1], dtype=theano.config.floatX), borrow=True)]


    def init_capsule_weight(self, n_hidden_l3, n_output):    

        return theano.shared(0.07*np.asarray(
                np.random.uniform(
                    low=-4 * np.sqrt(6. / n_output+n_hidden_l3),
                    high=4 * np.sqrt(6. / n_output+n_hidden_l3),
                    size=(n_hidden_l3, n_output)
                ), dtype=theano.config.floatX),borrow=True)



