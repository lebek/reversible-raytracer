import os
import numpy as np
from numpy.random import rand
from scene import *
from shader import *
from shape import *
import theano
from util import *

n = 100
x_dims = 64
y_dims = 64 


'''Returns initial scene information'''
def init_scene(x_dims, y_dims):

    #center1 = np.asarray([-0, -1.1, 4.5], dtype='float32')
    center1 = theano.shared(np.asarray([-4, -4, 20], dtype=theano.config.floatX), borrow=True)
    center2 = theano.shared(np.asarray([0, 0, 32], dtype=theano.config.floatX), borrow=True)
    
    material1 = Material((0.0, 0.9, 0.0), 0.3, 0.7, 0.5, 50.)
    material2 = Material((0.9, 0.0, 0.0), 0.3, 0.9, 0.4, 50.)
    
    transform1 = translate(center1) * scale((4., 4., 4.))
    transform2 = translate(center2) * scale((6, 6, 6))
    
    objects = [{'trans':transform1, 'material':material1}, \
               {'trans':transform2, 'material':material2}] 

    light_direction = (0., 0., 1.)
    light_intensity = (1., 1.,  1.)
    light_info = {'dir':light_direction, 'intensity':light_intensity}

    cam_loc1 = translate((0, 5,0))
    cam_loc2 = translate((0,-5,0))
    cam_dir1 = np.asarray([0,0,1], dtype='float32')   
    cam_dir2 = np.asarray([0,0,1], dtype='float32')   
    camera = [x_dims, y_dims, cam_loc1, cam_dir1, cam_loc2, cam_dir2]
    scene_info = {'light':light_info, 'objs':objects, 'camera':camera}

    return scene_info


'''Creates Scene Object given scene information'''
def make_scene(scene_info):

    #Shader & Camera setup
    x_dims, y_dims, cam_loc1, cam_dir1, cam_loc2, cam_dir2 = scene_info['camera']
    camera1 = Camera(x_dims, y_dims, cam_loc1, cam_dir1)
    camera2 = Camera(x_dims, y_dims, cam_loc2, cam_dir2)
    cameras = [camera1, camera2]
    shader = PhongShader()


    #Light
    light_direction = scene_info['light']['dir']
    light_intensity = scene_info['light']['intensity']
    light           = Light(light_direction, light_intensity)


    #Ojbects 
    shapes = []
    objs = scene_info['objs']
    num_objs = len(objs)

    #SPeH3R
    obj     = objs[0]
    trans   = obj['trans']
    material= obj['material']
    shapes.append(Sphere(trans, material))

    #SPeH3R
    obj     = objs[1]
    trans   = obj['trans']
    material= obj['material']
    shapes.append(Sphere(trans, material))

    scene1 = Scene(shapes, [light], camera1, shader)
    scene2 = Scene(shapes, [light], camera2, shader)
    scenes = [scene1, scene2]
    return scenes

LEFT    = 0
RIGHT   = 1

if __name__ == '__main__':

    scene_info  = init_scene(x_dims, y_dims)
    scenes      = make_scene(scene_info)
    image1      = scenes[LEFT].build()   
    image2      = scenes[RIGHT].build()   
    draw('./chocolate1.png', image1.eval())
    draw('./chocolate2.png', image2.eval())

    pass

