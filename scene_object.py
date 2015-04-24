import numpy as np
import theano.tensor as T
import theano

from scenemaker import *
from variable_set import *
from util import *

class SceneObject:
    def __init__(self, name):
        pass

class UnitSquare(SceneObject):
    def __init__(self, o2w, material):
        '''UnitSquare defined on the xy-plane, with vertices (0.5, 0.5, 0),
        (-0.5, 0.5, 0), (-0.5, -0.5, 0), (0.5, -0.5, 0), and normal (0, 0, 1).'''

        self.o2w = o2w
        self.w2o = o2w.inverse()
        self.material = material

    def _hit(self, rays, origin):
        mask_not_parallel_xy_plane = T.neq(rays[:,:,2],0)
        ts = -origin[2] / rays[:,:,2] #t is the
        mask_positive_t = T.gt(ts, 0)
        intersection = origin + ts.dimshuffle(0, 1, 'x')* rays
        mask_interior_of_unitsquare_x = T.gt(intersection[:,:,0], -0.5) * T.lt(intersection[:,:,0],0.5)
        mask_interior_of_unitsquare_y = T.gt(intersection[:,:,1], -0.5) * T.lt(intersection[:,:,1],0.5)
        mask_interior_of_unitsquare = mask_interior_of_unitsquare_x * mask_interior_of_unitsquare_y
        mask = mask_interior_of_unitsquare * mask_positive_t\
                        * mask_not_parallel_xy_plane

        all_falses = (1-mask)

        ts = ts * mask
        ts = T.switch(1-mask, float('inf'), ts)
        return mask, ts


    def distance(self, rayField):

        """Returns the distances along the rays that hits occur.
        If no hit, returns inf."""

        rf = self.w2o(rayField)
        mask, ts = self._hit(rf.rays, rf.origin)
        return ts #intersection

    def normals(self, rayField):
        rf = self.w2o(rayField)
        mask, ts = self._hit(rf.rays, rf.origin)
        mask_positive_t = T.gt(rf.origin[2], 0)

        pos_norm = T.concatenate([
            T.zeros_like(rf.rays[:,:,:2]),
            T.ones_like(rf.rays[:,:,:1])
        ], axis=2)
        neg_norm = T.concatenate([
            T.zeros_like(rf.rays[:,:,:2]),
            T.ones_like(rf.rays[:,:,:1])*-1
        ], axis=2)

        norm = pos_norm * mask_positive_t\
                            + neg_norm * ( 1-mask_positive_t)
        norm = norm * mask.dimshuffle(0,1,'x')
        return norm


class Sphere(SceneObject):
    def __init__(self, o2w, material):
        self.o2w = o2w
        self.w2o = o2w.inverse()
        self.material = material

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


    def surface_pts(self, rayField):

        rf = self.w2o(rayField)

        distance = self.distance(rayField)
        stabilized = T.switch(T.isinf(distance), 1000, distance)
        return rf.origin + (stabilized.dimshuffle(0, 1, 'x') * rays)


    def distance(self, rayField):
        """
        Returns the distances along the rays that hits occur.

        If no hit, returns inf.
        """

        rf = self.w2o(rayField)

        pdotv = T.tensordot(rf.rays, rf.origin, 1)
        vnorm = T.sum(rf.rays * rf.rays, axis=2)
        determinent = self._hit(rf.rays, rf.origin)
        distance1 = (- pdotv - T.sqrt(determinent)) / vnorm
        distance2 = (- pdotv + T.sqrt(determinent)) / vnorm
        distance = T.minimum(distance1, distance2)
        is_nan_or_negative = T.or_(determinent <= 0, T.isnan(determinent))
        stabilized = T.switch(is_nan_or_negative, float('inf'), distance)
        return stabilized

    def normals(self, rayField):
        """Returns the sphere normals at each hit point."""

        rf = self.w2o(rayField)

        distance = self.distance(rayField)
        distance = T.switch(T.isinf(distance), 0, distance)
        projections = (rf.origin) + (distance.dimshuffle(0, 1, 'x') * rf.rays)
        normals = projections / T.sqrt(
            T.sum(projections ** 2, 2)).dimshuffle(0, 1, 'x')
        return normals # need to fix
