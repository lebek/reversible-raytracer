import os
import numpy as np
from scenemaker import *

def simple_scene():
    material1 = Material('material 1', (0.2, 0.9, 0.4),
                         0.3, 0.7, 0.5, 50.)
    material2 = Material('material 2', (0.87, 0.1, 0.507),
                         0.3, 0.9, 0.4, 50.)
    material3 = Material('material 3', (0.2, 0.3, 1.),
                         0.3, 0.9, 0.4, 50.)

    objs = [
        Sphere('sphere 1', material1),
        Sphere('sphere 2', material2),
        #Sphere('sphere 3', material3),
        #Sphere('sphere 3', (5., -1., 1.), 1., material3)
        UnitSquare('square 1', material3)
    ]

    light = Light('light', (-1., -1., 2.), (1., 0.87, 0.961))
    #camera = Camera('camera', (0., 0., 0.), (1., 0., 0.), 128, 128)
    camera = Camera('camera', (0., 0., 0.), 128, 128)
    shader = PhongShader('shader')
    #shader = DepthMapShader('shader')
    return Scene(objs, [light], camera, shader)


def scene_setup0(scene):
    ''' Red sphere, Green egg (object order front to back)'''

    # t0 = translate(1,-1,10) * rotate('z', 90) * scala(1,2,1.5)
    # t1 = translate(1,1,6.)
    # s0.w2o = t1

    scene.translate(scene.objects[1], (1,1,6.))
    scene.translate(scene.objects[0], (1,-1,10))
    scene.rotate(scene.objects[0], 'z', 90.)
    scene.scale(scene.objects[0], (1,2,1.5), np.zeros((3,)))

def scene_setup1(scene):
    ''' Red sphere, Green egg on the top of blue carpet '''
    scene.translate(scene.objects[1], (1,1,6.))
    scene.translate(scene.objects[0], (1,-1,10))

    scene.rotate(scene.objects[0], 'z', 90.)
    scene.scale(scene.objects[0], (1,2,1.5), np.zeros((3,)))

    scene.translate(scene.objects[2], (-2,0,10))
    scene.rotate(scene.objects[2], 'y', 120)
    scene.scale(scene.objects[2], (7,7,7), np.zeros((3,)))
