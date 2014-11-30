import numpy as np
import theano.tensor as T
import theano
from copy import copy

# rays
L = T.dtensor3('L')

# origin of rays
o = T.dvector('o')

# center of sphere
c = T.dvector('c')
c2 = T.dvector('c2')

# radius of sphere
r = T.dscalar('r')
r2 = T.dscalar('r2')

# light
light = T.dvector('light')

def sphere_hit(center, radius):
    x = T.tensordot(L, (o - center), 1)
    decider = T.sqr(x) - T.dot(o - center, o - center) + T.sqr(radius)
    return decider

def sphere_dist(center, radius):
    x = T.tensordot(L, (o - center), 1)
    decider = sphere_hit(center, radius)
    distance = -x - T.sqrt(decider)
    distance_filter = T.switch(decider > 0, distance, float('inf'))
    distance_filter = T.switch(T.isnan(decider), float('inf'), distance_filter)
    return distance_filter

def sphere_normals(center, radius):
    distance = sphere_dist(center, radius)
    distance = T.switch(T.isinf(distance), 0, distance)
    sphere_projections = o + (distance.dimshuffle(0, 1, 'x') * L) - center
    normals = sphere_projections / T.sqrt(T.sum(T.mul(sphere_projections, sphere_projections), 2)).dimshuffle(0,1,'x')
    return normals

def diffuse_shading(c, r):
    shadings =  0.9*T.tensordot(sphere_normals(c, r), -light, 1)
    shadings_filter = T.switch(shadings >= 0, shadings, 0)
    return T.switch(T.isinf(sphere_dist(c, r)), 0, shadings_filter)

def phong_shading(c, r):
    diffuse_shadings = 0.9*T.tensordot(sphere_normals(c, r), -light, 1)
    normals = sphere_normals(c, r)
    rm = 2.0*(T.tensordot(normals, -light, 1).dimshuffle(0, 1, 'x'))*normals + light
    specular_shadings = 0.4*(T.tensordot(rm, [1., 0., 0.], 1) ** 20)
    shadings = diffuse_shadings + specular_shadings
    shadings_filter = T.switch(shadings >= 0, shadings, 0)
    # switch to color starts here
    colors = shadings_filter.dimshuffle(0, 1, 'x') * T.as_tensor_variable([1., 0., 0.]).dimshuffle('x', 'x', 0) * T.alloc(1., 512, 512, 1)
    return T.switch(T.isinf(sphere_dist(c, r).dimshuffle(0, 1, 'x')), [0., 0., 0.], colors)

# shadow of c2 on c1
def shadows(distances):
    surface_pts = o + (distances.dimshuffle(0, 1, 'x') * L)
    y = surface_pts - c2
    x = T.tensordot(y, -1*light, 1)
    decider = T.sqr(x) - T.sum(T.mul(y, y), 2) + T.sqr(r2)
    return decider

def shadows_and_shadings(center, radius):
    distance = sphere_dist(center, radius)
    dist_filt = T.switch(T.isinf(distance), 0, distance)
    # switch to color starts here
    with_shadows = T.switch(T.gt(shadows(dist_filt).dimshuffle(0, 1, 'x'), 0), [0., 0., 0.],
            phong_shading(center, radius))
    return with_shadows

intersect2 = T.switch(T.lt(sphere_dist(c, r).dimshuffle(0, 1, 'x'), sphere_dist(c2, r2).dimshuffle(0, 1, 'x')),
        shadows_and_shadings(c, r),
        phong_shading(c2, r2))

f = theano.function([L, o, c, r, c2, r2, light], intersect2, on_unused_input='ignore')

x_dims = 512
y_dims = 512
rays = np.dstack(np.meshgrid(np.linspace(0.5, -0.5, y_dims), np.linspace(-0.5, 0.5, x_dims), indexing='ij'))
rays = np.dstack([np.ones([y_dims, x_dims]), rays])
rays = np.divide(rays, np.linalg.norm(rays, axis=2).reshape(y_dims, x_dims, 1).repeat(3, 2))

light_dir = [2,-1,-1]
normalized_light =  light_dir/np.linalg.norm(light_dir)

cam_origin = [0., 0., 0.]
sphere_origin = [10., 0., 0.]
c2_o = [6., 1., 1.]
c2_r = 1.

pleasant_defaults = {
    'radius': 3.,
    'light': normalized_light,
    'sphere_origin': sphere_origin
}


def render(params):
    output = f(rays, cam_origin, params['sphere_origin'],
            params['radius'], c2_o, c2_r,  params['light'])
    return output


def optimize(params, x, y, lr, statusFn, maxIterations=30):
    curr_params = copy(params)

    #radius_fn = theano.function([L, o, c, r, c2, r2, light], T.grad(intersect2[y,x], r))
    statusFn('Optimizing light gradient function')
    light_fn = theano.function([L, o, c, r, c2, r2, light], T.grad(intersect2[y,x,0], light))

    statusFn('Optimizing sphere origin gradient function')
    sphere_origin_fn = theano.function([L, o, c, r, c2, r2, light], T.grad(intersect2[y,x,0], c))

    for i in range(maxIterations):
        statusPrefix = '%d of %d: ' % (i, maxIterations)
        loopStatus = lambda x: statusFn(statusPrefix + x)
        args = (rays, cam_origin, curr_params['sphere_origin'],
                curr_params['radius'], c2_o, c2_r, curr_params['light'])

        #radius_grad = radius_fn(*args)
        loopStatus('Computing light gradient')
        light_grad = light_fn(*args)

        loopStatus('Computing sphere origin gradient')
        sphere_origin_grad = sphere_origin_fn(*args)

        loopStatus('Updating parameters')
        #curr_params['curr_radius'] = curr_params['curr_radius'] +
        #    (lr * np.sign(curr_params['radius_grad']))
        curr_params['sphere_origin'] = curr_params['sphere_origin'] + (lr * sphere_origin_grad)
        curr_params['light'] = curr_params['light'] + (lr * light_grad)
        curr_params['light'] = curr_params['light'] / np.linalg.norm(curr_params['light'])
        print curr_params

        loopStatus('Rendering updated image')
        output = f(*args)

        yield (output, copy(curr_params))

    statusFn('Max iteration reached')
