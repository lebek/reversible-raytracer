import numpy as np
import theano.tensor as T
import theano
import copy

from scene_object import *
from variable_set import *
from util import *
from transform import *

def broadcasted_switch(a, b, c):
    return T.switch(a.dimshuffle(0, 1, 'x'), b, c)


class Shader:
    pass


class DepthMapShader(Shader):

    def __init__(self, maxDepth):
        self.maxDepth = maxDepth

    def shade(self, scene_object, lights, camera):
        distance = scene_object.distance(camera.rays)
        minv = 0
        maxv = self.maxDepth
        scaled = (distance - minv) / (maxv - minv)
        return (1 - scaled).dimshuffle(0, 1, 'x') * [1., 1., 1.]


class PhongShader(Shader):

    def __init__(self):
        pass

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

        return image


class Camera:
    def __init__(self, x_dims, y_dims):
        self.look_at = np.asarray([0,0,1.], dtype='float32')
        self.x_dims = x_dims
        self.y_dims = y_dims

    def make_rays(self, x_dims, y_dims, sampleDist_x=None, sampleDist_y=None):
        # this should be rewritten in theano - currently we can't do any
        # sensible optimization on camera parameters since we're calculating
        # the ray field prior to entering theano (thus losing parameterisation)

        rays = np.dstack(np.meshgrid(np.linspace(0.5, -0.5, y_dims),
                         np.linspace(-0.5, 0.5, x_dims), indexing='ij'))
        rays = np.dstack([rays, np.ones([y_dims, x_dims])])
        rays = np.divide(rays, np.linalg.norm(rays, axis=2).reshape(
                                        y_dims, x_dims, 1).repeat(3, 2))

        if sampleDist_x is not None: rays[:,:,0] = rays[:,:,0] + sampleDist_x / x_dims
        if sampleDist_y is not None: rays[:,:,1] = rays[:,:,1] + sampleDist_y / y_dims
        return RayField(T.as_tensor_variable([0., 0., 0.]), T.as_tensor_variable(rays))


class Light:
    def __init__(self, direction, intensity):
        self.direction = T.as_tensor_variable(direction)
        self.intensity = T.as_tensor_variable(intensity)

    def normed_dir(self):
        d = self.direction
        norm = T.sqrt(T.sqr(d[0]) + T.sqr(d[1]) + T.sqr(d[2]))
        return d/norm


class Material:
    def __init__(self, color, ks, kd, ka, shininess):
        """
        ks -- specular reflection constant
        kd -- diffuse reflection constant
        ka -- ambient reflection constant
        shininess -- shininess constant
        """
        self.ks = T.as_tensor_variable(ks)
        self.kd = T.as_tensor_variable(kd)
        self.ka = T.as_tensor_variable(ka)
        self.color = T.as_tensor_variable(color)
        self.shininess = T.as_tensor_variable(shininess)
