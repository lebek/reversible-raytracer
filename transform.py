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
        self.origin = T.as_tensor_variable(origin)
        self.rays = T.as_tensor_variable(directions)


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

def identity():
    """Returns the identity transform"""
    return Transform(np.asarray(np.eye(4, 4),dtype=theano.config.floatX) , np.asarray(np.eye(4, 4), dtype=theano.config.floatX))

def translate(x):
    """Returns a transform to represent a translation"""

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
    """Returns a transform to represent a scaling"""

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

def rotate(angle, axis):
    """Returns a transform to represent a rotation"""

    angle = T.as_tensor_variable(angle)
    axis = T.as_tensor_variable(axis)
    a = axis

    radians = angle*np.pi/180.0
    s = T.sin(radians)
    c = T.cos(radians)

    m = T.alloc(0., 4, 4)

    m = T.set_subtensor(m[0,0], a[0] * a[0] + (1. - a[0] * a[0]) * c)
    m = T.set_subtensor(m[0,1], a[0] * a[1] * (1. - c) - a[2] * s)
    m = T.set_subtensor(m[0,2], a[0] * a[2] * (1. - c) + a[1] * s)

    m = T.set_subtensor(m[1,0], a[0] * a[1] * (1. - c) + a[2] * s)
    m = T.set_subtensor(m[1,1], a[1] * a[1] + (1. - a[1] * a[1]) * c)
    m = T.set_subtensor(m[1,2], a[1] * a[2] * (1. - c) - a[0] * s)

    m = T.set_subtensor(m[2,0], a[0] * a[2] * (1. - c) - a[1] * s)
    m = T.set_subtensor(m[2,1], a[1] * a[2] * (1. - c) + a[0] * s)
    m = T.set_subtensor(m[2,2], a[2] * a[2] + (1. - a[2] * a[2]) * c)

    m = T.set_subtensor(m[3,3], 1)

    return Transform(m, m.T)
