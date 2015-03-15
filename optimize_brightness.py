import os
from matplotlib import pyplot as plt
from scenemaker import simple_scene
from grad_descent import GDOptimizer
from scipy.misc import imsave
import numpy
import theano

if not os.path.exists('output'):
    os.makedirs('output')

scene = simple_scene()
opt = GDOptimizer(scene)

print 'Rendering initial scene'
variables, values, image = scene.build()
render = theano.function(variables, image, on_unused_input='ignore')(*values)

def draw(fname, im):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im, interpolation='nearest')
    ax.add_patch(plt.Rectangle((85-3, 90-3), 6, 6, color='red', linewidth=2, fill=False))
    ax.add_patch(plt.Rectangle((90-3, 50-3), 6, 6, color='red', linewidth=2, fill=False))
    fig.savefig(fname, bbox_inches='tight', pad_inches=0)

draw('output/0.png', render)

print 'Building gradient functions'
optimization = opt.optimize(image[90, 85].sum()+image[50, 90].sum(), 0.02, 0.1, 90)

for i, update in enumerate(optimization):
    print 'Step', i+1
    draw('output/%d.png' % (i+1,), update)
