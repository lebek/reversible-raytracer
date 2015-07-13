import theano.tensor as T
from util import *


class Shader:
    pass


class DepthMapShader(Shader):

    def __init__(self, maxDepth):
        self.maxDepth = float(maxDepth)

    def shade(self, shape, lights, camera):
        distance = shape.distance(camera.rays)
        minv = 0
        maxv = self.maxDepth
        scaled = (distance - minv) / (maxv - minv)
        return (1 - scaled).dimshuffle(0, 1, 'x') *\
                        np.asarray([1., 1., 1.], dtype=theano.config.floatX)


class PhongShader(Shader):

    def __init__(self):
        pass

    def shade(self, shape, lights, camera):
        # See: http://en.wikipedia.org/wiki/Phong_reflection_model#Description

        # Since our material params are 1d we calculate bw shadings first and
        # convert to color after
        light = lights[0]
        material = shape.material
        normals = shape.normals(camera.rays)

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
        distances = shape.distance(camera.rays)
        return broadcasted_switch(T.isinf(distances), [0., 0., 0.], clipped)



