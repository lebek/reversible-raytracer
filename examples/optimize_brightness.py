import os
import numpy as np
from optimize import GDOptimizer
import theano
from util import *

### Hyper-parameters ###
OptmizeFlag=False
#-----------------------


if not os.path.exists('output'):
    os.makedirs('output')

center1 = theano.shared(np.asarray([-.5, -.5, 4], dtype=theano.config.floatX),
                       borrow=True)
center2 = theano.shared(np.asarray([.5, .5, 4], dtype=theano.config.floatX),
                       borrow=True)

material1 = Material((0.2, 0.9, 0.4), 0.3, 0.7, 0.5, 50.)
material2 = Material((0.87, 0.1, 0.507), 0.3, 0.9, 0.4, 50.)

t1 = translate(center1)
t2 = translate(center2) * rotate(90, (0, 0, 1)) * scale((1, 2, 1.5))

shapes = [
    Sphere(t1, material1),
    Sphere(t2, material2)
]

light = Light((-1., -1., 2.), (0.961, 1., 0.87))
camera = Camera(128, 128)
shader = PhongShader()
scene = Scene(shapes, [light], camera, shader)

opt = GDOptimizer()

print 'Rendering initial scene'
image = scene.build()
render_fn = theano.function([], image, on_unused_input='ignore')

drawWithMarkers('output/0.png', render_fn())

if OptmizeFlag:
    print 'Building gradient functions'
    train = opt.optimize([center1, center2],
        -image[90, 85].sum()-image[50, 90].sum(),
        0.0008, 0.1)


    for i in range(90):
        print 'Step', i+1
        print train()
        drawWithMarkers('output/%d.png' % (i+1,), render_fn())
