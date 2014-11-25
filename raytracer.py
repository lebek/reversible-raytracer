
import numpy as np
import theano.tensor as T
import theano

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
    distance_filter = T.switch(T.lt(0, decider), distance, float('inf'))
    return distance_filter

def sphere_normals(center, radius):
    distance = sphere_dist(center, radius)
    sphere_projections = o + (distance.dimshuffle(0, 1, 'x') * L) - center
    normals = sphere_projections / T.sqrt(T.sum(T.mul(sphere_projections, sphere_projections), 2)).dimshuffle(0,1,'x')
    return normals

def diffuse_shading(normals):
    shadings =  0.9*-T.tensordot(normals, light, 1)
    shadings_filter = T.switch(T.lt(0, shadings), shadings, 0)
    return shadings_filter

intersect = T.switch(T.lt(sphere_dist(c, r), sphere_dist(c2, r2)), 
        diffuse_shading(sphere_normals(c, r)), 
        diffuse_shading(sphere_normals(c2, r2)))
intersect2 =  T.switch(T.lt(0, intersect), intersect, 0)

f = theano.function([L, o, c, r, c2, r2, light], intersect, on_unused_input='ignore')

x_dims = 512
y_dims = 512
rays = np.dstack(np.meshgrid(np.linspace(0.5, -0.5, y_dims), np.linspace(-0.5, 0.5, x_dims), indexing='ij'))
rays = np.dstack([np.ones([y_dims, x_dims]), rays])
rays = np.divide(rays, np.linalg.norm(rays, axis=2).reshape(y_dims, x_dims, 1).repeat(3, 2))

light_dir = [2,-1,-1]
normalized_light =  light_dir/np.linalg.norm(light_dir)

cam_origin = [0., 0., 0.]
sphere_origin = [10., 0., 0.]

output = f(rays, cam_origin, sphere_origin, 3., [8., 2., 2.], 1.,  normalized_light)
print output

from matplotlib import pyplot as plt
from matplotlib import animation
from skimage.measure import block_reduce

fig = plt.figure()
ax = fig.add_subplot(111)
#ax.imshow(block_reduce(output, (4, 4), np.mean), interpolation='nearest').set_cmap('bone')
im = ax.imshow(output, interpolation='nearest')
im.set_cmap('bone')
im.set_clim(0.0,1.0)

def gd(x, y):
    acc = {
        'curr_radius': 2.,
        'curr_light': normalized_light,
        'curr_sphere_origin': sphere_origin,
        'radius_grad': 0,
        'light_grad': 0,
        'sphere_origin_grad': 0,
    }

    radius_fn = theano.function([L, o, c, r, light], T.grad(intersect[y,x], r))
    light_fn = theano.function([L, o, c, r, light], T.grad(intersect[y,x], light))
    sphere_origin_fn = theano.function([L, o, c, r, light], T.grad(intersect[y,x], c))

    def step(i, acc, radius_fn, light_fn, sphere_origin_fn):
        acc['radius_grad'] = radius_fn(rays, cam_origin, 
                acc['curr_sphere_origin'], acc['curr_radius'], acc['curr_light'])
        acc['light_grad'] = light_fn(rays, cam_origin, 
                acc['curr_sphere_origin'], acc['curr_radius'], acc['curr_light'])
        acc['sphere_origin_grad'] = sphere_origin_fn(rays, cam_origin, 
                acc['curr_sphere_origin'], acc['curr_radius'], acc['curr_light'])

        #acc['curr_radius'] = acc['curr_radius'] + (0.01 * np.sign(acc['radius_grad']))
        acc['curr_sphere_origin'] = acc['curr_sphere_origin'] + (0.008 * np.sign(acc['sphere_origin_grad']))
        acc['curr_light'] = acc['curr_light'] + (0.008 * np.sign(acc['light_grad']))
        acc['curr_light'] =  acc['curr_light']/np.linalg.norm(acc['curr_light'])
        print acc

        output = f(rays, cam_origin, acc['curr_sphere_origin'], acc['curr_radius'], acc['curr_light'])
        im.set_data(output)
        return im

    anim = animation.FuncAnimation(fig, step, frames=30,
            fargs=(acc, radius_fn, light_fn, sphere_origin_fn),
            interval=0)._start()

def onclick(event):
    print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
        event.button, event.x, event.y, event.xdata, event.ydata)
    #brighten_advise(int(event.xdata), int(event.ydata))
    gd(int(event.xdata), int(event.ydata))

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
