
import numpy as np
import theano.tensor as T
import theano

class GDOptimizer:
    
    def __init__(self, scene):
        self.vars, self.vals, self.image = scene.build()

    def _step(self, grad_fns, lr):
        for idx, var in enumerate(self.vars):
            grad = grad_fns[idx](*self.vals)
            if var.name == "ray field -> rays" or np.isnan(grad).any():
                print "skip grad"
                continue
            print var.name, grad
            self.vals[idx] = self.vals[idx] + lr * grad
            
            self.vals[idx] = np.array(self.vals[idx]).clip(
                var.low_bound, var.up_bound)

            print self.vals[idx]
            print ""
        
        return self.vals

    def optimize(self, loss, lr, statusFn, maxIterations=5):
        
        statusFn("Building gradient functions...")

        grad_fns = []
        for idx, var in enumerate(self.vars):
            grad_fns.append(
                theano.function(self.vars, T.grad(loss, var),
                                on_unused_input='ignore', allow_input_downcast=True))

        for i in range(maxIterations):
            statusFn("Step %d" % (i,))
            self.vals = self._step(grad_fns, lr)
            render = theano.function(self.vars, self.image, on_unused_input='ignore', 
                                     allow_input_downcast=True)(*self.vals)
            yield render
