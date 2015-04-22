
import numpy as np
import theano.tensor as T
import theano


class GDOptimizer:
    """
    Gradient Descent optimizer
    """

    def __init__(self, scene):
        self.vars, self.vals, self.image = scene.build()

    def locked(self, var):
        return ("ray field" in var.name or
                "material" in var.name or
                "intensity" in var.name or
                "camera" in var.name)

    def optimize(self, loss, lr, momentum=0):
        """
        Maximize the loss function with learning rate lr.
        """
        grads = T.grad(loss, self.vars)
        for var, gvar in zip(self.vars, grads):
            if self.locked(var):
                continue
            update_vars.append((var, var - lr * gvar))

        return theano.function([], loss, updates=update_vars)


