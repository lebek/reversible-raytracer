import os
import numpy as np
import theano.tensor as T
import theano
from scipy import misc

from linear_encoder import LinEncoder
from autoencoder_2ly import Autoencoder2ly
from variational_ae import VAE
from transform import *
from scene import *
from shader import *
from optimize import *



def scene(capsules, obj_params):

    shapes = []
    #TODO move the material information to attribute of capsule instance
    material1 = Material((0.0, 0.9, 0.0), 0.3, 0.7, 0.5, 50.)
    material2 = Material((0.9, 0.0, 0.0), 0.3, 0.9, 0.4, 50.)
    center2 = theano.shared(np.asarray([0, 0, 32], dtype=theano.config.floatX), borrow=True)

    for i in xrange(len(capsules)):

        capsule     = capsules[i]
        obj_param   = obj_params[i]
        t1 = translate(obj_param) #* scale(obj_param[1,:])
        if capsule.name == 'sphere':
            shapes.append(Sphere(t1 * scale((4, 4, 4)), material1))
        elif capsule.name == 'square':
            shapes.append(Square(t1, material1))
        elif capsule.name == 'light':
            shapes.append(Light(t1, material1))

    shapes.append(Sphere(translate(center2) * scale((6, 6, 6)), material2))
    light = Light((-0., -0., 1), (1., 1., 1.)) # (0.961, 1., 0.87)
    camera = Camera(img_sz, img_sz)
    shader = PhongShader()
    #shader = DepthMapShader()

    scene = Scene(shapes, [light], camera, shader)
    return scene.build()


def test_1image(num_capsule  = 1,
                epsilon      = 0.00001,
                epsilon_adam = 0.0001,
                num_epoch    = 5000,
                opt_image    = './orbit_dataset/40.png'):

    if not os.path.exists('./output/one_imgs'):
        os.makedirs('./output/one_imgs')

    train_data = np.asarray([misc.imread(opt_image).flatten()], dtype='float32')/255.0
    N,D = train_data.shape
    train_data = theano.shared(train_data)

    global img_sz
    if RGBflag:
        img_sz = int(np.sqrt(D/3))
    else:
        img_sz = int(np.sqrt(D))

    ae = LinEncoder(scene, D, 300,  num_capsule)
    #ae = Autoencoder(scene, D, 300, 30, 10, num_capsule)
    opt = MGDAutoOptimizer(ae)
   
    train_ae = opt.optimize(train_data)
    train_aeADAM = opt.optimizeADAM(train_data)
    get_recon = theano.function([], ae.get_reconstruct(train_data[0]))
    get_center= theano.function([], ae.encoder(train_data[0].dimshuffle('x',0))[0].flatten())
    
    recon = get_recon()
    center = get_center()
    
    imsave('output/one_imgs/test_balls0.png', recon)
    print '...Initial center1 (%g,%g,%g)' % (center[0], center[1], center[2])
    print recon.sum()
    
    n=0;
    while (n<num_epoch):
    
        n+=1
        eps = get_epsilon(epsilon, num_epoch, n)
        train_loss  = train_ae(0, eps)
    
        if n % 50 ==0 or n < 5:
            center = get_center()
            print '...Epoch %d Eps %g, Train loss %g, Center (%g, %g, %g)' \
                    % (n, eps, train_loss, center[0], center[1], center[2])
    
            image = get_recon()
            imsave('output/one_imgs/test_balls%d.png' % (n,), image)
  

def test_2images(epsilon,
               epsilon_adam = 0.0001,
               num_epoch    = 6000,
               num_capsule  = 1,
               ae_type      = 'vae'):

    if not os.path.exists('./output/two_imgs'):
        os.makedirs('./output/two_imgs')

    data = np.load('orbit_dataset.npz')['arr_0'] / 255.0
    data = data.astype('float32')
    train_data = data[0:2,:,:,:] 
    N,D,D,K = train_data.shape
    train_data = theano.shared(train_data.reshape(N, D*D*K))
    global img_sz 
    img_sz = D 

    #ae = LinEncoder(scene, img_sz*img_sz*3, 300,  num_capsule)
    ae = Autoencoder2ly(scene, img_sz*img_sz*3, 600, 30, num_capsule)
    #if ae_type == 'vae':
    #    ae = VAE(scene, img_sz*img_sz*3, 300, 30, 10, num_capsule)
    #else:
    #    ae = Autoencoder(scene, img_sz*img_sz*3, 300, 30, 10, num_capsule)
    opt = MGDAutoOptimizer(ae)

    train_ae = opt.optimize(train_data)
    train_aeADAM = opt.optimizeADAM(train_data)

    get_recon1 = theano.function([], ae.get_reconstruct(train_data[0]))
    get_recon2 = theano.function([], ae.get_reconstruct(train_data[1]))
    get_center= theano.function([], ae.encoder(train_data[0].dimshuffle('x',0))[0].flatten())

    center = get_center()
    imsave('output/two_imgs/1_test_balls0.png', get_recon1())
    imsave('output/two_imgs/2_test_balls0.png', get_recon2())
    print '...Initial center1 (%g,%g,%g)' % (center[0], center[1], center[2])

    n=0;
    while (n<num_epoch):
        n+=1
        eps = get_epsilon(epsilon, num_epoch, n)
        train_loss = 0 
        for i in xrange(2):
            train_loss += train_ae(i, eps)
    
        if n % 100 == 0 or n < 5:
            center = get_center()
            print '...Epoch %d Eps %g, Train loss %g, Center (%g, %g, %g)' \
                    % (n, eps, train_loss, center[0], center[1], center[2])
    
            imsave('output/two_imgs/1_test_balls%d.png' % (n,), get_recon1())
            imsave('output/two_imgs/2_test_balls%d.png' % (n,), get_recon2())
 
    pass
   
    
if __name__ == '__main__':

    global RGBflag
    RGBflag = True
    A = 0

    if A:
        test_1image()
    else: 
        ae_type      = 'lae'
        if ae_type=='vae': #Note: training with VAE doesn't work yet
            epsilon      = 0.0000001 
        elif ae_type=='lae':
            epsilon      = 0.00002
        else:
            epsilon      = 0.0000002
        test_2images(epsilon, ae_type=ae_type)
    
