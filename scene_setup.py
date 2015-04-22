import os
import numpy as np
from scenemaker import simple_scene


def scene_setup0(scene):
    ''' Red sphere, Green egg (object order front to back)'''

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
