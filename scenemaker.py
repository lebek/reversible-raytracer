import numpy as np
import theano.tensor as T
import theano
import copy


def broadcasted_switch(a, b, c):
    return T.switch(a.dimshuffle(0, 1, 'x'), b, c)


class VariableSet:
    """Holds metadata about variables used in theano functions"""

    def __init__(self, name):
        self.name = name
        self.variables = []
        self.children = []

    def _as_var(self, value, name):
        return theano.shared(value=np.asarray(value), name=name, borrow=True)

    def add(self, value, label, low_bound=-np.inf, up_bound=np.inf):
        name = '%s -> %s' % (self.name, label)
        variable = self._as_var(value, name)
        variable.low_bound = low_bound
        variable.up_bound = up_bound
        self.variables.append((variable, value))
        return variable

    def add_child(self, child):
        self.children.append(child)

    def get(self):
        variables = copy.copy(self.variables)
        for c in self.children:
            variables.extend(c.get())
        return variables


class Shader:
    pass


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

    def build(self):
        # returns top-level render function and associated variables
        image = T.alloc(0, self.camera.x_dims, self.camera.y_dims, 3)
        min_dists = T.alloc(float('inf'), self.camera.x_dims, self.camera.y_dims)

        # for each object find its shadings and draw closer objects on top
        for obj in self.objects:
            dists = obj.distance(self.camera.rays)
            shadings = self.shader.shade(obj, self.lights, self.camera)

            # TODO: for each object != obj, draw shadow of object on obj
            #for obj2 in self.objects:
            #    if obj == obj2: continue
            #    shadings = broadcasted_switch(obj2.shadow(
            #        obj.surface_pts(self.camera.rays), self.lights) < 0,
            #                                  shadings, [0., 0., 0.])

            image = broadcasted_switch(dists < min_dists, shadings, image)
            min_dists = T.switch(dists < min_dists, dists, min_dists)

        variables, values = self.gather_varvals()

        return variables, values, image

    def render(self):
        variables, values, image = self.build()
        f = theano.function(variables, image, on_unused_input='ignore')
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
        # this should be rewritten in theano - currently we can't do any
        # sensible optimization on camera parameters since we're calculating
        # the ray field prior to entering theano (thus losing parameterisation)
        rays = np.dstack(np.meshgrid(np.linspace(0.5, -0.5, y_dims),
                         np.linspace(-0.5, 0.5, x_dims), indexing='ij'))
        rays = np.dstack([np.ones([y_dims, x_dims]), rays])
        rays = np.divide(rays, np.linalg.norm(rays, axis=2).reshape(
            y_dims, x_dims, 1).repeat(3, 2))
        return RayField('ray field', self.position, rays)


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


class SceneObject:
    def __init__(self, name):
        pass


class Sphere(SceneObject):
    def __init__(self, name, center, radius, material):
        self.variables = VariableSet(name)
        self.center = self.variables.add(center, 'center')
        self.radius = self.variables.add(radius, 'radius', 0)
        self.material = material
        self.variables.add_child(material.variables)

    def _hit(self, ray_field):
        x = ray_field.origin - self.center
        y = T.tensordot(ray_field.rays, x, 1)
        determinent = T.sqr(y) - T.dot(x, x) + T.sqr(self.radius)
        return determinent

    def shadow(self, points, lights):
        """
        Returns whether points are in shadow of this object.

        See: http://en.wikipedia.org/wiki/Line-sphere_intersection
        """
        y = points - self.center # vector from points to our center
        x = T.tensordot(y, -1*lights[0].normed_dir(), 1)
        decider = T.sqr(x) - T.sum(T.mul(y, y), 2) + T.sqr(self.radius)

        # if shadow, below is >= 0
        is_nan_or_nonpos = T.or_(T.isnan(decider), decider <= 0)
        return T.switch(is_nan_or_nonpos, 1, -x - T.sqrt(decider))

    def surface_pts(self, ray_field):
        distance = self.distance(ray_field)
        stabilized = T.switch(T.isinf(distance), 1000, distance)
        return ray_field.origin + (stabilized.dimshuffle(0, 1, 'x') * ray_field.rays)

    def distance(self, ray_field):
        """
        Returns the distances along the rays that hits occur.

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


def simple_scene():
    material1 = Material('material 1', (0.2, 0.9, 0.4),
                         0.8, 0.7, 0.5, 40.)
    material2 = Material('material 2', (0.87, 0.1, 0.507),
                         0.8, 0.9, 0.4, 60.)
    material3 = Material('material 3', (0.2, 0.3, 1.),
                         0.8, 0.9, 0.4, 60.)

    objs = [
        Sphere('sphere 1', (10., 0., -1.), 2., material1),
        Sphere('sphere 2', (6., 1., 1.), 1., material2),
        Sphere('sphere 3', (5., -1., 1.), 1., material3)
    ]

    light = Light('light', (2., -1., -1.), (0.87, 0.961, 1.))
    camera = Camera('camera', (0., 0., 0.), (1., 0., 0.), 128, 128)
    shader = PhongShader('shader')
    return Scene(objs, [light], camera, shader)
