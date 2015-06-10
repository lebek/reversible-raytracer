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

        get_grad = theano.function([], grads[0], givens={X:train_data[0]})
        get_gradb = theano.function([], grads[1], givens={X:train_data[0]})
        return opt, get_grad, get_gradb



from scipy import misc
train_data = [misc.imread('output/0.jpg')[:,:,0].flatten().astype('float32')/255.0]
#train_data = [misc.imread('output/0.png')[:,:,0].flatten().astype('float32')/255.0]

#from scipy import ndimage
#train_data = [ndimage.imread('output/0.jpg', mode='RGB')[:,:,0].flatten().astype('float32')/255.0]


#center1 = theano.shared(np.asarray([-.5, -.5, 4], dtype=theano.config.floatX),
#                       borrow=True)

def scene(center1):

    material1 = Material((0.2, 0.9, 0.4), 0.3, 0.7, 0.5, 50.)

    t1 = translate(center1)

    shapes = [
        Sphere(t1, material1)
    ]

    light = Light((-1., -1., 2.), (0.961, 1., 0.87))
    camera = Camera(128, 128)
    shader = PhongShader()
    scene = Scene(shapes, [light], camera, shader)
    return scene.build()

ae = Autoencoder(scene, 128*128, 100, 10)
import pdb; pdb.set_trace()
opt = MGDAutoOptimizer(ae)
train_ae, get_grad, get_gradb = opt.optimize(train_data, 0.01)

get_recon = theano.function([], ae.get_reconstruct(train_data[0]))
get_centre = theano.function([], ae.encoder(train_data[0]))
get_cost  = theano.function([], ae.cost(train_data[0]))

n=0;
center_i =get_centre()
print '...Epoch %d Train loss %g, Centre (%g, %g, %g)' \
                    % (n, get_cost(),center_i[0], center_i[1], center_i[2])

while (n<1000):
    n+=1
    train_loss = train_ae()
    center_i =get_centre()
    print '...Epoch %d Train loss %g, Centre (%g, %g, %g)' \
                    % (n, train_loss,center_i[0], center_i[1], center_i[2])
#    ggg =get_grad()
#    gbb =get_gradb()
    #import pdb; pdb.set_trace()
    image = get_recon()
    imsave('test%d.png' % (n,), image)



