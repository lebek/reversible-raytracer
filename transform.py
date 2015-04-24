import numpy as np
import theano
import theano.tensor as T


class Point():
    def __init__(self, p):
        self.p = p


class PointField():
    def __init__(self, pf):
        self.pf = pf


class VectorField():
    def __init__(self, vf):
        self.vf = vf


class RayField():
    def __init__(self, origin, directions):
        self.origin = origin
        self.rays = directions


class Transform():
    def __init__(self, m, mInv):
        self.m = m
        self.mInv = mInv

    def inverse(self):
        return Transform(self.mInv, self.m)

    def __mul__(self, other):
        m = T.dot(self.m, other.m)
        mInv = T.dot(other.mInv, self.mInv)
        return Transform(m, mInv)

    def __call__(self, x):
        if isinstance(x, RayField):
            o = x.origin
            r = x.rays
            origin = T.dot(self.m, [o[0], o[1], o[2], 1])[:3]
            rays = T.concatenate([r, T.zeros_like(r)[:, :, :1]], axis=2)
            rays = T.tensordot(self.m, rays, [1, 2]).T[:, :, :3]
            return RayField(origin, rays)

        #if isinstance(x, Point):
        #    return Point(T.dot(self.m, [x.p[0], x.p[1], x.p[2], 1])[:3])
        #elif isinstance(x, Vector):
        #    return Vector(T.dot(self.m, [x.v[0], x.v[1], x.v[2], 0])[:3])
        #elif isinstance(x, Ray):
        #    return Ray(self(x.o), self(x.d))


def translate(x):
    """Returns a transform matrix to represent a translation"""

    x = T.as_tensor_variable(x)

    m = T.eye(4, 4)
    m = T.set_subtensor(m[0,3], x[0])
    m = T.set_subtensor(m[1,3], x[1])
    m = T.set_subtensor(m[2,3], x[2])

    mInv = T.eye(4, 4)
    mInv = T.set_subtensor(mInv[0,3], -x[0])
    mInv = T.set_subtensor(mInv[1,3], -x[1])
    mInv = T.set_subtensor(mInv[2,3], -x[2])

    return Transform(m, mInv)


def scale(x):
    """Creates a transform matrix to represent a translation"""

    x = T.as_tensor_variable(x)

    m = T.eye(4, 4)
    m = T.set_subtensor(m[0,0], x[0])
    m = T.set_subtensor(m[1,1], x[1])
    m = T.set_subtensor(m[2,2], x[2])

    mInv = T.eye(4, 4)
    mInv = T.set_subtensor(mInv[0,0], 1./x[0])
    mInv = T.set_subtensor(mInv[1,1], 1./x[1])
    mInv = T.set_subtensor(mInv[2,2], 1./x[2])

    return Transform(m, mInv)
