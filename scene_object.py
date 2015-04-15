import numpy as np
import theano.tensor as T
import theano

from scenemaker import *
from variable_set import *
from util import *

class SceneObject:
    def __init__(self, name):
        pass

    def translate(point, factor):

        point[0] = point[0] + factor[0];
        point[0] = point[1] + factor[1];
        point[0] = point[2] + factor[2];
        return point

    def scale(point, factor):

        point[0] = point[0] * factor[0];
        point[0] = point[1] * factor[1];
        point[0] = point[2] * factor[2];
        return point

    def rotate(point, angle, axis='x'):

        toRadian = 2 * np.pi/360.0;
        if axis=='x':
            point[1] = np.cos(angle * toRadian) * point[1] - np.sin(angle *toRadian)*point[2];
            point[2] = np.sin(angle * toRadian) * point[1] + np.cos(angle *toRadian)*point[2];
        elif axis=='y':
           point[0] = np.cos(angle * toRadian) * point[0] - np.sin(angle *toRadian)*point[2];
           point[2] = np.sin(angle * toRadian) * point[0] + np.cos(angle *toRadian)*point[2];
        elif axis=='z':
            point[1] = np.cos(angle * toRadian) * point[1] - np.sin(angle *toRadian)*point[2];
            point[2] = np.sin(angle * toRadian) * point[1] + np.cos(angle *toRadian)*point[2];
        return point


class UnitSquare(SceneObject):
    def __init__(self, name, material):
        '''UnitSquare defined on the xy-plane, with vertices (0.5, 0.5, 0),
        (-0.5, 0.5, 0), (-0.5, -0.5, 0), (0.5, -0.5, 0), and normal (0, 0, 1).'''

        self.variables = VariableSet(name)
        self.material = material
        self.variables.add_child(material.variables)
        #self.transform = self.variables.add(np.eye((3)), 'modelToWorldProj')
        #self.translate = self.variables.add(np.zeros((3,)), 'modelToWorldTrans')
        self.invtransform = self.variables.add(np.eye((3)), 'worldToModelProj')
        self.invtranslate = self.variables.add(np.zeros((3,)), 'worldToModelTrans')

    def _hit(self, rays, origin):


        mask_not_parallel_xy_plane = T.neq(rays[:,:,2],0)
        ts = -ray_field.origin[2] / rays[:,:,2] #t is the
        mask_positive_t = T.gt(ts, 0)
        intersection = ray_field.origin + ts.dimshuffle(0, 1, 'x')* rays
        mask_interior_of_unitsquare = T.gt(intersection, -0.5) * T.lt(intersection,0.5)
        mask = mask_interior_of_unitsquare * mask_positive_t.dimshuffle(0,1,'x')\
                        * mask_not_parallel_xy_plane.dimshuffle(0,1,'x')

        all_falses = (1-mask)
        intersection = T.switch(all_falses, float('inf'), intersection)
        intersection = T.set_subtensor(intersection[:,:,2], T.zeros_like(rays[:,:,2]))
        intersection = T.dot(self.trans, intersection)
        return mask, intersection,ts


    def distance(self, ray_field):

        """Returns the distances along the rays that hits occur.
        If no hit, returns inf."""

        rays = T.dot(self.invtransform, ray_field.rays)
        origin = T.dot(self.invtransform, ray_field.origin)  + self.invtranslate
        mask, intersection, ts = self._hit(ray, origin)

        return ts #intersection

    def normals(self, ray_field):

        mask, intersection,ts = self._hit(ray_field)
        mask_positive_t = T.gt(ts, 0)

        pos_norm = np.tile( np.asarray([0.0,0.0,1.0]), \
                 [ray_field.x_dims,ray_field.y_dims,1])
        neg_norm = np.tile( np.asarray([0.0,0.0,-1.0]), \
                 [ray_field.x_dims,ray_field.y_dims,1])

        norm = pos_norm * mask_positive_t.dimshuffle(0,1,'x') \
                            + neg_norm * ( 1-mask_positive_t).dimshuffle(0,1,'x')
        norm = norm * mask

        return transNorm(self.invtransform, norm)


class Sphere(SceneObject):
    def __init__(self, name, material):
        self.variables = VariableSet(name)
        self.material = material
        self.variables.add_child(material.variables)
        #self.transform = self.variables.add(np.eye((3)), 'modelToWorldProj')
        #self.translate = self.variables.add(np.zeros((3,)), 'modelToWorldTrans')
        self.invtransform = self.variables.add(np.eye((3), dtype='float32'), 'worldToModelProj')
        self.invtranslate = self.variables.add(np.zeros((3,), dtype='float32'), 'worldToModelTrans')

    def _hit(self, rays, origin):
        pnorm = T.dot(origin, origin)
        vnorm = T.sum(rays * rays, axis=2)
        pdotv = T.tensordot(rays, origin, 1)
        determinent = T.sqr(pdotv) - vnorm * (pnorm - 1)
        return determinent

    def shadow(self, points, lights):
        """
        Returns whether points are in shadow of this object.

        See: http://en.wikipedia.org/wiki/Line-sphere_intersection
        """
        y = points  # vector from points to our center
        x = T.tensordot(y, -1*lights[0].normed_dir(), 1)
        decider = T.sqr(x) - T.sum(T.mul(y, y), 2) + 1

        # if shadow, below is >= 0
        is_nan_or_nonpos = T.or_(T.isnan(decider), decider <= 0)
        return T.switch(is_nan_or_nonpos, -1, -x - T.sqrt(decider))


    def surface_pts(self, ray_field):

        origin  = T.dot(self.invtransform, ray_field.origin) + self.invtranslate
        rays    = T.dot(ray_field.rays, self.invtransform.T)

        distance = self.distance(ray_field)
        stabilized = T.switch(T.isinf(distance), 1000, distance)
        return origin + (stabilized.dimshuffle(0, 1, 'x') * rays)


    def distance(self, ray_field):
        """
        Returns the distances along the rays that hits occur.

        If no hit, returns inf.
        """
        origin  = T.dot(self.invtransform, ray_field.origin) + self.invtranslate
        rays    = T.dot(ray_field.rays, self.invtransform.T)

        pdotv = T.tensordot(rays, origin, 1)
        vnorm = T.sum(rays * rays, axis=2)
        determinent = self._hit(rays, origin)
        distance = (- pdotv - T.sqrt(determinent)) / vnorm
        is_nan_or_negative = T.or_(determinent <= 0, T.isnan(determinent))
        stabilized = T.switch(is_nan_or_negative, float('inf'), distance)
        return stabilized

    def normals(self, ray_field):
        """Returns the sphere normals at each hit point."""

        origin  = T.dot(self.invtransform, ray_field.origin) + self.invtranslate
        rays    = T.dot(ray_field.rays, self.invtransform.T)

        distance = self.distance(ray_field)
        distance = T.switch(T.isinf(distance), 0, distance)
        projections = (origin) + (distance.dimshuffle(0, 1, 'x') * rays)
        normals = projections / T.sqrt(
            T.sum(projections ** 2, 2)).dimshuffle(0, 1, 'x')
        return transNorm(self.invtransform, normals)
