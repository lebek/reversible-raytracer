import os
import numpy as np
from scenemaker import simple_scene
from grad_descent import GDOptimizer
import theano
from util import *
from scene_setup import *

### Hyper-parameters ###
OptmizeFlag=False
#-----------------------


if not os.path.exists('output'):
    os.makedirs('output')

scene = simple_scene()
scene_setup1(scene)


opt = GDOptimizer(scene)

print 'Rendering initial scene'
variables, values, image = scene.build()
render_fn = theano.function([], image, on_unused_input='ignore')

drawWithMarkers('output/0.png', render_fn())


if OptmizeFlag:
    print 'Building gradient functions'
    train = opt.optimize(-image[90, 85].sum()-image[50, 90].sum(),
                         0.0008, 0.1)
    
    
    for i in range(90):
        print 'Step', i+1
        print train()
        drawWithMarkers('output/%d.png' % (i+1,), render_fn())




