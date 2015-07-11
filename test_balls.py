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

train_data = np.array([misc.imread('example.png').flatten()], dtype='float32')/255.0
N,D = train_data.shape
img_sz = int(np.sqrt(D))


def scene(capsules, obj_params):

    shapes = []
    #TODO move the material information to attribute of capsule instance 
    material1 = Material((0.2, 0.9, 0.4), 0.3, 0.7, 0.5, 50.)
    for i in xrange(len(capsules)):
        
        capsule     = capsules[i]
        obj_param   = obj_params[i]
        t1 = translate(obj_param[:3]) * scale(obj_param[3:])
        if capsule.name == 'sphere':
            shapes.append(Sphere(t1, material1))
        elif capsule.name == 'square':
            shapes.append(Square(t1, material1))
        elif capsule.name == 'light':
            shapes.append(Light(t1, material1))

    light = Light((-1., -1., 2.), (0.961, 1., 0.87))
    camera = Camera(img_sz, img_sz)
    shader = PhongShader()
    scene = Scene(shapes, [light], camera, shader)
    return scene.build()

ae = Autoencoder(scene, D, 300, 30, 10)
opt = MGDAutoOptimizer(ae)

#recon = ae.get_reconstruct(train_data[0])[:,:,0].eval()
#imsave('output/test_balls0.png', recon)

epsilon = 0.0001
num_epoch = 200

train_ae = opt.optimize(train_data)
get_recon = theano.function([], ae.get_reconstruct(train_data[0])[:,:,0])
get_center= theano.function([], ae.encoder(train_data[0]))

recon = get_recon()
center = get_center()[0]
imsave('output/test_balls0.png', recon)
print '...Initial center1 (%g,%g,%g)' % (center[0], center[1], center[2])
print recon.sum()

n=0;
while (n<num_epoch):
    n+=1
    eps = get_epsilon(epsilon, num_epoch, n)
    train_loss  = train_ae(eps)
    center      = get_center()[0]
    print '...Epoch %d Train loss %g, Center (%g, %g, %g)' \
                    % (n, train_loss, center[0], center[1], center[2])

    if n % 10 ==0:
        image = get_recon()
        imsave('output/test_balls%d.png' % (n,), image)






