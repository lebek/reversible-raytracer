import numpy as np
import theano.tensor as T
import theano
import copy

from scene_object import *
from variable_set import *
from util import *

def broadcasted_switch(a, b, c):
    return T.switch(a.dimshuffle(0, 1, 'x'), b, c)


class Shader:
    pass


class DepthMapShader(Shader):

    def __init__(self, name):
        self.variables = VariableSet(name)

    def shade(self, scene_object, lights, camera):
        distance = scene_object.distance(camera.rays)
        minv = T.min(distance)
        maxv = T.max(T.switch(T.isinf(distance), minv, distance))
        scaled = (distance - minv) / (maxv - minv)
        return (1 - scaled).dimshuffle(0, 1, 'x') * [1., 1., 1.]

class PhongShader(Shader):

    def __init__(self, name):
        self.variables = VariableSet(name)

    def shade(self, scene_object, lights, camera):
        # See: http://en.wikipedia.org/wiki/Phong_reflection_model#Description

        # Since our material params are 1d we calculate bw shadings first and
        # convert to color after
        light = lights[0]
        material = scene_object.material
        normals = scene_object.normals(camera.rays)

        ambient_light = material.ka

        # diffuse (lambertian)
        diffuse_shadings = material.kd*T.tensordot(normals, -light.normed_dir(), 1)

        # specular
        rm = 2.0*(T.tensordot(normals, -light.normed_dir(), 1).dimshuffle(
            0, 1, 'x'))*normals + light.normed_dir()
        specular_shadings = material.ks*(T.tensordot(rm, camera.look_at, 1) ** material.shininess)

        # phong
        phong_shadings = ambient_light + diffuse_shadings + specular_shadings

        colorized = phong_shadings.dimshuffle(0, 1, 'x') * material.color.dimshuffle('x', 'x', 0) * light.intensity.dimshuffle('x', 'x', 0)
        clipped = T.clip(colorized, 0, 1)
        distances = scene_object.distance(camera.rays)
        return broadcasted_switch(T.isinf(distances), [0., 0., 0.], clipped)


class Scene:
    def __init__(self, objects, lights, camera, shader):
        self.objects = objects
        self.lights = lights
        self.camera = camera
        self.shader = shader

    def gather_varvals(self):
        # collect all the theano variables from the scene graph
        # -> we need it all in one place in order to call the top-level render
        # function
        varvals = []

        for o in self.objects:
            varvals.extend(o.variables.get())

        for l in self.lights:
            varvals.extend(l.variables.get())

        varvals.extend(self.camera.variables.get())
        varvals.extend(self.shader.variables.get())

        variables = [v[0] for v in varvals]
        values = [v[1] for v in varvals]
        return (variables, values)

    def build(self, antialias_samples=4):

        # returns top-level render function and associated variables
        image = T.alloc(0., self.camera.x_dims, self.camera.y_dims, 3)

        #Anti-Aliasing
        sampleDist_x = np.random.random((self.camera.x_dims, self.camera.y_dims,antialias_samples))
        sampleDist_y = np.random.random((self.camera.x_dims, self.camera.y_dims,antialias_samples))

        for sample in xrange(antialias_samples): #TODO USE SCAN 

            #Make Rays
            self.camera.rays = self.camera.make_rays(self.camera.x_dims, self.camera.y_dims,\
                            sampleDist_x=(sampleDist_x[:,:,sample] + sample)/antialias_samples,
                            sampleDist_y=(sampleDist_y[:,:,sample] + sample)/antialias_samples)
            #self.camera.variables.add_child(self.camera.rays.variables)
            image_per_sample = T.alloc(0.0, self.camera.x_dims, self.camera.y_dims, 3)
            min_dists = T.alloc(float('inf'), self.camera.x_dims, self.camera.y_dims)

            # for each object find its shadings and draw closer objects on top
            for obj in self.objects:
                dists = obj.distance(self.camera.rays)
                shadings = self.shader.shade(obj, self.lights, self.camera)
                #for each object != obj, draw shadow of object on obj
                #for obj2 in self.objects:
                #    if obj == obj2: continue
                #    shadings = broadcasted_switch(obj2.shadow(
                #        obj.surface_pts(self.camera.rays), self.lights) < 0, shadings, [0., 0., 0.])
                image_per_sample = broadcasted_switch(dists < min_dists, shadings, image_per_sample)
                min_dists = T.switch(dists < min_dists, dists, min_dists)

            image = image + image_per_sample
        image = image / antialias_samples
        variables, values = self.gather_varvals()

        return variables, values, image

    def render(self):
        variables, values, image = self.build()
        f = theano.function(variables, image, on_unused_input='ignore')
        return f(*values)

    def translate(self, sceneobject, trans):

        translate = np.zeros((3,), dtype='float32');
        translate[0] = trans[0];
        translate[1] = trans[1];
        translate[2] = trans[2];
        #sceneobject.translate.set_value(translate)
        translate[0] = -trans[0];
        translate[1] = -trans[1];
        translate[2] = -trans[2];
        sceneobject.invtranslate.set_value(translate)

    def scale(self, sceneobject, sc, origin):

        scaleM = np.eye((3), dtype='float32'); scaleT = np.zeros((3,), dtype='float32')
        scaleM[0,0] = sc[0];   #scaleT[0] = origin[0] - sc[0] * origin[0]
        scaleM[1,1] = sc[1];   #scaleT[1] = origin[1] - sc[1] * origin[1]
        scaleM[2,2] = sc[2];   #scaleT[2] = origin[2] - sc[2] * origin[2]

        scaleM[0,0] = 1./sc[0];  scaleT[0] = origin[0] - 1./sc[0] * origin[0]
        scaleM[1,1] = 1./sc[1];  scaleT[1] = origin[1] - 1./sc[1] * origin[1]
        scaleM[2,2] = 1./sc[2];  scaleT[2] = origin[2] - 1./sc[2] * origin[2]
        sceneobject.invtransform.set_value(np.dot(scaleM, sceneobject.invtransform.get_value()))
        sceneobject.invtranslate.set_value(np.dot(scaleM, \
                        sceneobject.invtranslate.get_value())+ scaleT)


    def rotate(self, sceneobject, axis, angle):

        rotateM = np.eye((3), dtype='float32'); rotateT = np.ones((3,), dtype='float32')
        toRadian = 2*np.pi/360.0;

        for i in xrange(2):
            if axis=='x':
                rotateM[0,0] = 1;
                rotateM[1,1] = np.cos(angle*toRadian);
                rotateM[1,2] = -np.sin(angle*toRadian);
                rotateM[2,1] = np.sin(angle*toRadian);
                rotateM[2,2] = np.cos(angle*toRadian);
            elif axis=='y':
                rotateM[0,0] = np.cos(angle*toRadian);
                rotateM[1,1] = np.sin(angle*toRadian);
                rotateM[1,2] = 1;
                rotateM[2,1] = -np.sin(angle*toRadian);
                rotateM[2,2] = np.cos(angle*toRadian);
            elif axis=='z':
                rotateM[0,0] = np.cos(angle*toRadian);
                rotateM[1,1] = -np.sin(angle*toRadian);
                rotateM[1,2] = np.sin(angle*toRadian)
                rotateM[2,1] = np.cos(angle*toRadian);
                rotateM[2,2] = 1

            if i == 0:
                #sceneobject.trans = np.dot(sceneobject.trans, rotateM)
                angle = -angle;
            else:
                sceneobject.invtransform.set_value(np.dot(rotateM, sceneobject.invtransform.get_value()))
                sceneobject.invtranslate.set_value(np.dot(rotateM, \
                                sceneobject.invtranslate.get_value())+ rotateT)




class RayField:
    def __init__(self, name, origin, rays, x_dims, y_dims):
        self.variables = VariableSet(name)
        self.origin = origin
        self.rays = rays
        self.rays = self.variables.add(rays, 'rays')
        self.x_dims = x_dims
        self.y_dims = y_dims

class Camera:
    def __init__(self, name, position, look_at, x_dims, y_dims):
        self.variables = VariableSet(name)
        self.position = self.variables.add(position, 'position')
        self.look_at = look_at
        self.x_dims = x_dims
        self.y_dims = y_dims

    def make_rays(self, x_dims, y_dims, sampleDist_x=None, sampleDist_y=None):
        # this should be rewritten in theano - currently we can't do any
        # sensible optimization on camera parameters since we're calculating
        # the ray field prior to entering theano (thus losing parameterisation)
        rays = np.dstack(np.meshgrid(np.linspace(0.5, -0.5, y_dims),
                         np.linspace(-0.5, 0.5, x_dims), indexing='ij'))
        rays = np.dstack([np.ones([y_dims, x_dims]), rays])
        rays = np.divide(rays, np.linalg.norm(rays, axis=2).reshape(
                                        y_dims, x_dims, 1).repeat(3, 2))
        
        if sampleDist_x is not None: rays[:,:,1] = rays[:,:,1] + sampleDist_x / x_dims 
        if sampleDist_y is not None: rays[:,:,2] = rays[:,:,2] + sampleDist_y / y_dims 
        return RayField('ray field', self.position, rays, x_dims, y_dims)


class Light:
    def __init__(self, name, direction, intensity):
        self.variables = VariableSet(name)
        self.direction = self.variables.add(direction, 'direction')
        self.intensity = self.variables.add(intensity, 'intensity', 0, 1)

    def normed_dir(self):
        d = self.direction
        norm = T.sqrt(T.sqr(d[0]) + T.sqr(d[1]) + T.sqr(d[2]))
        return d/norm


class Material:
    def __init__(self, name, color, ks, kd, ka, shininess):
        """
        ks -- specular reflection constant
        kd -- diffuse reflection constant
        ka -- ambient reflection constant
        shininess -- shininess constant
        """
        self.variables = VariableSet(name)
        self.ks = self.variables.add(ks, 'ks', 0, 1)
        self.kd = self.variables.add(kd, 'kd', 0, 1)
        self.ka = self.variables.add(ka, 'ka', 0, 1)
        self.color = self.variables.add(color, 'color', 0, 1)
        self.shininess = self.variables.add(shininess, 'shininess', 0, 1)


def simple_scene():
    material1 = Material('material 1', (0.2, 0.9, 0.4),
                         0.8, 0.7, 0.5, 40.)
    material2 = Material('material 2', (0.87, 0.1, 0.507),
                         0.8, 0.9, 0.4, 60.)
    material3 = Material('material 3', (0.2, 0.3, 1.),
                         0.8, 0.9, 0.4, 60.)

    objs = [
        Sphere('sphere 1', material1),
        Sphere('sphere 2', material2),
        #Sphere('sphere 3', material3),
        #Sphere('sphere 3', (5., -1., 1.), 1., material3)
        #UnitSquare('square 1', material2)
    ]

    light = Light('light', (2., -1., -1.), (0.87, 0.961, 1.))
    camera = Camera('camera', (0., 0., 0.), (1., 0., 0.), 128, 128)
    shader = PhongShader('shader')
    #shader = DepthMapShader('shader')
    return Scene(objs, [light], camera, shader)

