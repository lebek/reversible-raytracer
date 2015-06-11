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

        get_grad = theano.function([], grads[3], givens={X:train_data[0]})
        get_gradb = theano.function([], grads[-1], givens={X:train_data[0]})
        return opt, get_grad, get_gradb



from scipy import misc
#train_data = [misc.imread('15.jpg').flatten().astype('float32')/255.0]
train_data = [misc.imread('15.png').flatten().astype('float32')/255.0]

#from scipy import ndimage
#train_data = [ndimage.imread('output/0.jpg', mode='RGB')[:,:,0].flatten().astype('float32')/255.0]


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

ae = Autoencoder(scene, 32*32, 300, 30, 10)
opt = MGDAutoOptimizer(ae)
train_ae, get_grad, get_gradb = opt.optimize(train_data, 0.01)

get_recon = theano.function([], ae.get_reconstruct(train_data[0]))
get_centre1 = theano.function([], ae.encoder(train_data[0])[0])
get_centre2 = theano.function([], ae.encoder(train_data[0])[1])
get_cost  = theano.function([], ae.cost(train_data[0]))

n=0;
center_i1 =get_centre1()
center_i2 =get_centre2()
print '...Epoch %d Train loss %g, Center1 (%g, %g, %g), Center1 (%g, %g, %g)' \
                    % (n, get_cost(),center_i1[0], center_i1[1], center_i1[2],\
                                     center_i2[0], center_i2[1], center_i2[2])
print '...Epoch %d Train loss %g, ' % (n, get_cost()) 

while (n<5):
    n+=1

    ggg =get_grad()
    gbb =get_gradb()
    import pdb; pdb.set_trace()

    train_loss = train_ae()
    center_i1 =get_centre1()
    center_i2 =get_centre2()
    print '...Epoch %d Train loss %g, Center1 (%g, %g, %g), Center1 (%g, %g, %g)' \
                    % (n, get_cost(),center_i1[0], center_i1[1], center_i1[2],\
                                     center_i2[0], center_i2[1], center_i2[2])

    #print '...Epoch %d Train loss %g, ' \
    #            % (n, train_loss) 
    image = get_recon()
    imsave('test%d.png' % (n,), image)


