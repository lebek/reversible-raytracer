import os
import numpy as np
import theano.tensor as T
import theano
from scipy import misc

from autoencoder import Autoencoder
from transform import *
from scene import *
from shader import *
from optimize import *

if not os.path.exists('output'):
    os.makedirs('output')


#Hyper-parameters
RGBflag      = True
num_capsule = 1
epsilon = 0.0002
num_epoch = 20


#train_data = np.array([misc.imread('example.png').flatten()], dtype='float32')/255.0
train_data = np.asarray([misc.imread('1.png').flatten()], dtype='float32')/255.0
N,D = train_data.shape
if RGBflag:
    img_sz = int(np.sqrt(D/3))
else:
    img_sz = int(np.sqrt(D))


def scene(capsules, obj_params):

    shapes = []
    #TODO move the material information to attribute of capsule instance
    material1 = Material((0.2, 0.9, 0.4), 0.3, 0.7, 0.5, 50.)
    material2 = Material((0.87, 0.1, 0.507), 0.3, 0.9, 0.4, 50.)
    center2 = theano.shared(np.asarray([0, 0, 8], dtype=theano.config.floatX), borrow=True)

    for i in xrange(len(capsules)):

        capsule     = capsules[i]
        obj_param   = obj_params[i]
        t1 = translate(obj_param) #* scale(obj_param[1,:])
        if capsule.name == 'sphere':
            shapes.append(Sphere(t1, material1))
        elif capsule.name == 'square':
            shapes.append(Square(t1, material1))
        elif capsule.name == 'light':
            shapes.append(Light(t1, material1))

    shapes.append(Sphere(translate(center2) * scale((1.5, 1.5, 1.5)), material2))
    light = Light((-1., -1., 2.), (0.961, 1., 0.87))
    camera = Camera(img_sz, img_sz)
    if RGBflag:
        shader = PhongShader()
    else:
        shader = DepthMapShader()

    scene = Scene(shapes, [light], camera, shader)
    return scene.build()

ae = Autoencoder(scene, D, 300, 30, 10, num_capsule)
opt = MGDAutoOptimizer(ae)

train_ae = opt.optimize(train_data)
get_recon = theano.function([], ae.get_reconstruct(train_data[0]))
get_center= theano.function([], ae.encoder(train_data[0])[0].flatten())

recon = get_recon()
center = get_center()

imsave('output/test_balls0.png', recon)
#print '...Initial center1 (%g,%g,%g), radius (%g,%g,%g)' \
#                        % (center[0], center[1], center[2], center[3], center[4], center[5])
print '...Initial center1 (%g,%g,%g)' % (center[0], center[1], center[2])
print recon.sum()

n=0;
while (n<num_epoch):

    n+=1
    eps = get_epsilon(epsilon, num_epoch, n)
    train_loss  = train_ae(eps)
    center      = get_center()
    print '...Epoch %d Train loss %g, Center (%g, %g, %g)' \
                % (n, train_loss, center[0], center[1], center[2])

    #cbias = ae.capsules[0].params[1].get_value()
    #print '...cBias (%g, %g, %g)' % (cbias[0], cbias[1], cbias[2])

    if n % 2 ==0 or n < 4:
        image = get_recon()
        imsave('output/test_balls%d.png' % (n,), image)




