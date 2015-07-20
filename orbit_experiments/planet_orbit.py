import os
import numpy as np
from numpy.random import rand
from scene import *
from shader import *
from shape import *
import theano
from util import *

n = 100
x_dims = 32
y_dims = 32

# Generates n x_dims-by-y_dims image samples containing 2 spheres with
# randomly assigned centers. Saves the result in dataset.npz

if not os.path.exists('orbit_dataset'):
    os.makedirs('orbit_dataset')


material1 = Material((0.0, 0.9, 0.0), 0.3, 0.7, 0.5, 50.)
material2 = Material((0.9, 0.0, 0.0), 0.3, 0.9, 0.4, 50.)

center1 = theano.shared(np.asarray([0, 0, 32], dtype=theano.config.floatX),
                        borrow=True)
center2 = theano.shared(np.asarray([0, 0, 32], dtype=theano.config.floatX),
                        borrow=True)

shapes = [
    Sphere(translate(center1) * scale((4., 4., 4.)), material1),
    Sphere(translate(center2) * scale((6, 6, 6)), material2)
]

light = Light((0., 0., 1.), (1., 1.,  1.))
camera = Camera(x_dims, y_dims)
#shader = DepthMapShader(6.1)
shader = PhongShader()
scene = Scene(shapes, [light], camera, shader)

image = scene.build()
render_fn = theano.function([], image, on_unused_input='ignore')

def random_orbit_position(v):
   
    x=1;y=1;
    z = 32 
    x = (1 if rand() > 0.5 else -1) * rand()
    y = (1 if rand() > 0.5 else -1) * np.sqrt(1.0 - x**2) 
    x = x * 3 * 4  
    y = y * 3 * 4 
    v.set_value(np.asarray([x, y, z], dtype=theano.config.floatX))
    return (x,y,z)

targets = np.zeros((n, 3), dtype=theano.config.floatX)
dataset = np.zeros((n, x_dims, y_dims, 3), dtype=np.uint8)
for i in range(n):
    target = random_orbit_position(center1)
    #random_transform(center2)
    render = render_fn()
    dataset[i] = (render * 255).astype(np.uint8)
    targets[i] = target
    draw('orbit_dataset/%d.png' % (i,), render)

np.savez('orbit_dataset', dataset)
np.savez('orbit_target', target)


