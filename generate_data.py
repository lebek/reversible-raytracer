import os
import numpy as np
from numpy.random import rand
from scenemaker import *
import theano
from util import *

if not os.path.exists('dataset'):
    os.makedirs('dataset')

material1 = Material('material 1', (0.2, 0.9, 0.4),
                     0.3, 0.7, 0.5, 50.)
material2 = Material('material 2', (0.87, 0.1, 0.507),
                     0.3, 0.9, 0.4, 50.)

objs = [
    Sphere('sphere 1', material1),
    Sphere('sphere 2', material2)
]

light = Light('light', (2., -1., -1.), (0.87, 0.961, 1.))
camera = Camera('camera', (0., 0., 0.), (1., 0., 0.), 128, 128)
shader = DepthMapShader('shader', 6)
scene = Scene(objs, [light], camera, shader)

variables, values, image = scene.build()
render_fn = theano.function([], image, on_unused_input='ignore')

def random_transform(obj):
    scene.translate(obj, (rand()*2+4, rand()*4-2, rand()*4-2))
    #scene.scale(obj, (rand()+1, rand()+1, rand()+1), np.zeros((3,)))

for i in range(100):
    random_transform(scene.objects[0])
    random_transform(scene.objects[1])
    draw('dataset/%d.png' % (i,), render_fn())
