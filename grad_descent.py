
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

    def optimize(self, loss, lr, momentum=0, maxIterations=30):
        """
        Maximize the loss function with learning rate lr.
        """

        render_fn = theano.function([], self.image,
                                    on_unused_input='ignore',
                                    allow_input_downcast=True)

        # grad_fns = []
        # for idx, var in enumerate(self.vars):
        #     if self.locked(var):
        #         grad_fns.append([])
        #         continue
        #     grad_fns.append(
        #         theano.function(self.vars, T.grad(loss, var),
        #                         on_unused_input='ignore',
        #                         allow_input_downcast=True))

        update_vars=[]
        grads = T.grad(loss, self.vars)
        for var, gvar in zip(self.vars, grads):
            if self.locked(var): continue
            #if 'center' not in var: continue
            update_vars.append((var, var - lr * gvar))

        train = theano.function([], loss, updates=update_vars)
        return train, render_fn



        # prev_updates = [0] * len(self.vars)
        # for step in range(maxIterations):
        #     for idx, var in enumerate(self.vars):
        #         if self.locked(var): continue
        #         grad = grad_fns[idx](*self.vals)
        #         if np.isnan(grad).any():
        #             print "nan grad"
        #             continue
        #         print var.name, grad
        #         update = (lr * grad) + (momentum * prev_updates[idx])
        #         self.vals[idx] = self.vals[idx] + update
        #         self.vals[idx] = np.array(self.vals[idx]).clip(
        #             var.low_bound, var.up_bound)

        #         prev_updates[idx] = update

        #         print self.vals[idx]
        #         print ""

        #     yield render_fn()
