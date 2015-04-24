import numpy as np
import theano.tensor as T
import theano


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
