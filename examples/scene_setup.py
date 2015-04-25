import os
import numpy as np
from scene import *
from shape import *
from shader import *

def scene1():
    ''' Red sphere, Green egg (object order front to back)'''
    material1 = Material((0.2, 0.9, 0.4), 0.3, 0.7, 0.5, 50.)
    material2 = Material((0.87, 0.1, 0.507), 0.3, 0.9, 0.4, 50.)

    t1 = translate((-1, 1, 6.))
    t2 = translate((-1, -1, 10)) * rotate(90, (0,0,1)) * scale((1,2,1.5))
    shapes = [
        Sphere(t1, material1),
        Sphere(t2, material2)
    ]

    light = Light((-1., -1., 2.), (0.961, 1., 0.87))
    camera = Camera(128, 128)
    shader = PhongShader()
    return Scene(shapes, [light], camera, shader)

def scene2():
    ''' Red sphere, Green egg on the top of blue carpet '''
    material1 = Material((0.2, 0.9, 0.4), 0.3, 0.7, 0.5, 50.)
    material2 = Material((0.87, 0.1, 0.507), 0.3, 0.9, 0.4, 50.)
    material3 = Material((0.2, 0.1, 0.807), 0.3, 0.9, 0.4, 50.)

    t1 = translate((1, 1, 6.))
    t2 = translate((1, -1, 10)) * rotate(90, (0, 0, 1)) * scale((1, 2, 1.5))
    t3 = translate((1, 0, 10)) * rotate(-50, [1., 0., 0.]) * scale((7, 7, 7))
    shapes = [
        Sphere(t1, material1),
        Sphere(t2, material2),
        Square(t3, material3)
    ]

    light = Light((-1., -1., 2.), (0.961, 1., 0.87))
    camera = Camera(128, 128)
    shader = PhongShader()
    return Scene(shapes, [light], camera, shader)
