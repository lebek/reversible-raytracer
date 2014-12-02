import numpy as np
import theano.tensor as T
import theano


def broadcasted_switch(a, b, c):
    return T.switch(a.dimshuffle(0, 1, 'x'), b, c)


class VariableSet:

    def __init__(self, name):
        self.name = name
        self.variables = []
        self.children = []

    def _as_var(self, value, name):
        return T.as_tensor_variable(value).type.make_variable(name)

    def add(self, value, label):
        name = '%s -> %s' % (self.name, label)
        variable = self._as_var(value, name)
        self.variables.append((variable, value))
        return variable

    def add_child(self, child):
        self.children.append(child)

    def get(self):
        variables = self.variables
        for c in self.children:
            variables.extend(c.get())
        return variables


class Shader:
    pass


class PhongShader(Shader):

    def __init__(self, name):
        self.variables = VariableSet(name)

    def shade(self, scene_object, lights, camera):
        light = lights[0]
        material = scene_object.material
        normals = scene_object.normals(camera.rays)

        ambient_light = 0.1

        # diffuse (lambertian)
        diffuse_shadings = material.kd*T.tensordot(normals, -light.direction, 1)

        # specular
        rm = 2.0*(T.tensordot(normals, -light.direction, 1).dimshuffle(
            0, 1, 'x'))*normals + light.direction
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

    def render(self):
        image = T.alloc(0, self.camera.x_dims, self.camera.y_dims, 3)
        min_dists = T.alloc(float('inf'), self.camera.x_dims, self.camera.y_dims)
        for obj in self.objects:
            dists = obj.distance(self.camera.rays)
            shadings = self.shader.shade(obj, self.lights, self.camera)
            image = broadcasted_switch(dists < min_dists, shadings, image)
            min_dists = T.switch(dists < min_dists, dists, min_dists)

        variables, values = self.gather_varvals()
        f = theano.function(variables, image, on_unused_input='ignore')

        #for v in variables:
        #    try:
        #        print theano.function(variables, T.grad(image[300, 300], v),
        #                            on_unused_input='ignore')(*values)
        #    except:
        #        print 'failed to find gradient for', v

        return f(*values)


class RayField:
    def __init__(self, name, origin, rays):
        self.variables = VariableSet(name)
        self.origin = origin
        self.rays = rays
        self.rays = self.variables.add(rays, 'rays')


class Camera:
    def __init__(self, name, position, look_at, x_dims, y_dims):
        self.variables = VariableSet(name)
        self.position = self.variables.add(position, 'position')
        self.look_at = look_at
        self.x_dims = x_dims
        self.y_dims = y_dims
        self.rays = self.make_rays(x_dims, y_dims)
        self.variables.add_child(self.rays.variables)

    def make_rays(self, x_dims, y_dims):
        rays = np.dstack(np.meshgrid(np.linspace(0.5, -0.5, y_dims),
                         np.linspace(-0.5, 0.5, x_dims), indexing='ij'))
        rays = np.dstack([np.ones([y_dims, x_dims]), rays])
        rays = np.divide(rays, np.linalg.norm(rays, axis=2).reshape(
            y_dims, x_dims, 1).repeat(3, 2))
        return RayField('ray field', self.position, rays)


class Light:
    def __init__(self, name, direction, intensity):
        self.variables = VariableSet(name)
        self.direction = self.variables.add(
            direction/np.linalg.norm(direction), 'direction')
        self.intensity = self.variables.add(intensity, 'intensity')


class Material:
    def __init__(self, name, color, ks, kd, ka, shininess):
        """
        ks -- specular reflection constant
        kd -- diffuse reflection constant
        ka -- ambient reflection constant
        shininess -- shininess constant
        """
        self.variables = VariableSet(name)
        self.ks = self.variables.add(ks, 'ks')
        self.kd = self.variables.add(kd, 'kd')
        self.ka = self.variables.add(ka, 'ka')
        self.color = self.variables.add(color, 'color')
        self.shininess = self.variables.add(shininess, 'shininess')


class SceneObject:
    def __init__(self, name):
        pass


class Sphere(SceneObject):
    def __init__(self, name, center, radius, material):
        self.variables = VariableSet(name)
        self.center = self.variables.add(center, 'center')
        self.radius = self.variables.add(radius, 'radius')
        self.material = material
        self.variables.add_child(material.variables)

    def _hit(self, ray_field):
        x = ray_field.origin - self.center
        y = T.tensordot(ray_field.rays, x, 1)
        determinent = T.sqr(y) - T.dot(x, x) + T.sqr(self.radius)
        return determinent

    def distance(self, ray_field):
        """Returns the distances along the rays that hits occur.

        If no hit, returns inf.
        """
        x = ray_field.origin - self.center
        y = T.tensordot(ray_field.rays, x, 1)
        determinent = self._hit(ray_field)
        distance = -y - T.sqrt(determinent)
        is_nan_or_negative = T.or_(determinent <= 0, T.isnan(determinent))
        stabilized = T.switch(is_nan_or_negative, float('inf'), distance)
        return stabilized

    def normals(self, ray_field):
        """Returns the sphere normals at each hit point."""
        distance = self.distance(ray_field)
        distance = T.switch(T.isinf(distance), 0, distance)
        x = ray_field.origin - self.center
        projections = x + (distance.dimshuffle(0, 1, 'x') * ray_field.rays)
        normals = projections / T.sqrt(
            T.sum(projections ** 2, 2)).dimshuffle(0, 1, 'x')
        return normals

material1 = Material('material 1', (1., 1., 1.), 0.0, 0.9, 0., 20.)
material2 = Material('material 2', (0.87, 0., 0.507), 0.5, 0.9, 0., 20.)
material3 = Material('material 3', (0., 0., 1.), 0.5, 0.9, 0., 20.)

objs = [
    Sphere('sphere 1', (10., 0., -1.), 2., material1),
    Sphere('sphere 2', (6., 1., 1.), 1., material2),
    Sphere('sphere 3', (5., -1., 1.), 1., material3)
]

light = Light('light', (2., -1., -1.), (0.87, 0.961, 1.))
camera = Camera('camera', (0., 0., 0.), (1., 0., 0.), 512, 512)
shader = PhongShader('shader')
scene = Scene(objs, [light], camera, shader)

image = scene.render()

import scipy
scipy.misc.imsave('render.png', image)
