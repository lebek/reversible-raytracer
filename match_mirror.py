import os
import numpy as np
from grad_descent import GDOptimizer
import theano
from util import *
from transform import *
from scenemaker import *

if not os.path.exists('output'):
    os.makedirs('output')


tv = lambda x: T.as_tensor_variable(np.array(x))

center1 = theano.shared(np.asarray([-.5, -.5, 3], dtype=theano.config.floatX),
                       borrow=True)
center2 = theano.shared(np.asarray([.5, .5, 3], dtype=theano.config.floatX),
                       borrow=True)

material1 = Material((0.2, 0.9, 0.4), 0.3, 0.7, 0.5, 50.)
material2 = Material((0.87, 0.1, 0.507), 0.3, 0.9, 0.4, 50.)

objs = [
    Sphere(translate(center1), material1),
    Sphere(translate(center2), material2),
    UnitSquare(translate((0, 0, 2)), material2)
]

light = Light((-1., -1., 2.), (1., 0.87, 0.961))
camera = Camera(128, 128)
shader = PhongShader()
scene = Scene(objs, [light], camera, shader)

print 'Rendering initial scene'
image = scene.build()
render_fn = theano.function([], image, on_unused_input='ignore')

render = render_fn()
flipped = np.fliplr(render)
draw('output/0.png', render)
draw('output/0lr.png', flipped)


cost = ((image - flipped) ** 2).sum()

print 'Building gradient functions'
train = GDOptimizer().optimize([center1, center2], cost, 0.000008, 0.1)

for i in range(90):
    print 'Step', i+1
    print train()
    draw('output/%d.png' % (i+1,), render_fn())
