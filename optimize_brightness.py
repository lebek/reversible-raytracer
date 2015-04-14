import os
import numpy as np
from matplotlib import pyplot as plt
from scenemaker import simple_scene
from grad_descent import GDOptimizer
import theano

if not os.path.exists('output'):
    os.makedirs('output')

scene = simple_scene()
scene.objects[0].trans,scene.objects[0].invtrans \
            = scene.scale(scene.objects[0], (1.1,1.1,1.1), np.zeros((3,)))

opt = GDOptimizer(scene)

print 'Rendering initial scene'
variables, values, image = scene.build()
render = theano.function([], image, on_unused_input='ignore')()

def draw(fname, im):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im, interpolation='nearest')
    ax.add_patch(plt.Rectangle((85-3, 90-3), 6, 6, color='red',
                               linewidth=2, fill=False))
    ax.add_patch(plt.Rectangle((90-3, 50-3), 6, 6, color='red',
                               linewidth=2, fill=False))
    fig.savefig(fname, bbox_inches='tight', pad_inches=0)

draw('output/0.png', render)

print 'Building gradient functions'
train, render = opt.optimize(-image[90, 85].sum()-image[50, 90].sum(),
                             0.02, 0.1, 90)

#theano.function([], image[90, 85].sum()+image[50, 90].sum()

for i in range(90):
    print 'Step', i+1
    print train()
    draw('output/%d.png' % (i+1,), render())
