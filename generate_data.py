import os
import numpy as np
from numpy.random import rand
from scenemaker import *
import theano
from util import *

n = 10
x_dims = 128
y_dims = 128

# Generates n x_dims-by-y_dims image samples containing 2 spheres with
# randomly assigned centers. Saves the result in dataset.npz

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

light = Light('light', (-1., -1., 2.), (0.961, 1., 0.87))
camera = Camera('camera', (0., 0., 0.), x_dims, y_dims)
shader = DepthMapShader('shader', 8)
scene = Scene(objs, [light], camera, shader)

variables, values, image = scene.build()
render_fn = theano.function([], image, on_unused_input='ignore')

def random_transform(obj):
    scene.translate(obj, (0, 0, 6))
    scene.translate(obj, (rand()*2-1, rand()*2-1, rand()*2-1))
    #scene.scale(obj, (rand()+1, rand()+1, rand()+1), np.zeros((3,)))

dataset = np.zeros((n, x_dims, y_dims), dtype=np.uint8)
for i in range(n):
    random_transform(scene.objects[0])
    random_transform(scene.objects[1])
    render = render_fn()[:, :, 0]
    dataset[i] = (render * 255).astype(np.uint8)
    draw('dataset/%d.png' % (i,), render)

np.savez('dataset', dataset)
