import numpy as np
import theano.tensor as T
import theano


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

    def __init__(self, name, ks, kd, ka):
        """
        ks -- specular reflection constant
        kd -- diffuse reflection constant
        ka -- ambient reflection constant
        """
        self.variables = VariableSet(name)
        self.ks = self.variables.add(ks, 'ks')
        self.kd = self.variables.add(kd, 'kd')
        self.ka = self.variables.add(ka, 'ka')

    def shade(self, scene_object, lights, camera):
        light = lights[0]
        normals = scene_object.normals(camera.rays)
        shadings = T.clip(0.9*T.tensordot(normals, -light.direction, 1), 0, 1)
        distances = scene_object.distance(camera.rays)
        return T.switch(T.isinf(distances), 0, shadings)


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
        image = T.alloc(0, self.camera.x_dims, self.camera.y_dims)
        min_dists = T.fill(image, float('inf'))
        for obj in self.objects:
            dists = obj.distance(self.camera.rays)
            shadings = self.shader.shade(obj, self.lights, self.camera)
            image = T.switch(dists < min_dists, shadings, image)
            min_dists = T.switch(dists < min_dists, dists, min_dists)

        variables, values = self.gather_varvals()
        f = theano.function(variables, image, on_unused_input='ignore')
        return f(*values)


class RayField:
    def __init__(self, name, origin, rays):
        self.variables = VariableSet(name)
        self.origin = origin
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
    def __init__(self, name, color, shininess):
        self.variables = VariableSet(name)
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

material = Material('material', (1, 0, 0), 20)

objs = [
    Sphere('sphere 1', (10, 0, 0), 3, material),
    Sphere('sphere 2', (6, 1, 1), 1, material),
    Sphere('sphere 3', (5, -1, 1), 1, material)
]

light = Light('light', (2, -1, -1), 1)
camera = Camera('camera', (0, 0, 0), (1, 0, 0), 512, 512)
shader = PhongShader('shader', 1, 1, 1)
scene = Scene(objs, [light], camera, shader)

image = scene.render()

import scipy
scipy.misc.imsave('render.png', image)
