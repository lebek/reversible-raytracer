import unittest
import numpy as np
from transform import *

class TestTransform(unittest.TestCase):
    def setUp(self):
        pass

    def test_rotate(self):
        t = rotate(20, (0, 0, 1))
        f = theano.function([], t.m)
        cmp = np.isclose(f(), np.array([
            [0.93969262, -0.34202015,  0.,  0.],
            [0.34202015,  0.93969262,  0.,  0.],
            [0.,  0.,  1.,  0.],
            [0.,  0.,  0.,  1.]
        ]))
        self.assertTrue(np.all(cmp))

    def test_composition(self):
        t = translate((4, 5, 6)) * rotate(20, (0, 0, 1))
        f = theano.function([], t.m)
        cmp = np.isclose(f(), np.array([
            [0.93969262, -0.34202015,  0.,  4.],
            [0.34202015,  0.93969262,  0.,  5.],
            [0.,  0.,  1.,  6.],
            [0.,  0.,  0.,  1.]
        ]))
        self.assertTrue(np.all(cmp))

    def test_apply(self):
        r = RayField((1, 0, 0), np.tile([0, 1, 0], (10, 10, 1)))
        t = translate((4, 5, 6)) * rotate(90, (0, 0, 1))

        result = theano.function([], t(r).origin)()
        cmpOrigin = np.isclose(result, np.array([4, 6, 6]))
        self.assertTrue(np.all(cmpOrigin))

        result = theano.function([], t(r).rays)()
        cmpRays = np.isclose(result, np.tile([-1, 0, 0], (10, 10, 1)))
        self.assertTrue(np.all(cmpRays))
