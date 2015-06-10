import theano
import theano.tensor as T
import numpy as np

class Autoencoder():

    # each render_var gets its own l2 layer
    def __init__(self, scene, n_visible,
                 n_hidden_l1, n_hidden_l2):
        self.scene = scene
        self.n_visible = n_visible
        self.n_hidden_l1 = n_hidden_l1
        self.n_hidden_l2 = n_hidden_l2

        vis_to_l1 = np.asarray(
            np.random.uniform(
                low=-4 * np.sqrt(6. / (n_visible + n_hidden_l1)),
                high=4 * np.sqrt(6. / (n_visible + n_hidden_l1)),
                size=(n_visible, n_hidden_l1)
            ),
            dtype=theano.config.floatX
        )

        l1_to_l2 = np.asarray(
            np.random.uniform(
                low=-4 * np.sqrt(6. / (n_hidden_l1 + n_hidden_l2)),
                high=4 * np.sqrt(6. / (n_hidden_l1 + n_hidden_l2)),
                size=(n_hidden_l1, n_hidden_l2)
            ),
            dtype=theano.config.floatX
        )

        #self.l1_biases = theano.shared(np.zeros(n_hidden_l1), borrow=True)
        #self.l2_biases = theano.shared(np.zeros(n_hidden_l2), borrow=True)
        self.rvar_biases = theano.shared(np.zeros(3), borrow=True)

        #self.vis_to_l1 = theano.shared(vis_to_l1, name="vis_to_l1", borrow=True)
        #self.l1_to_l2 = theano.shared(l1_to_l2, name="l1_to_l2", borrow=True)

        self.l2_to_rvar = theano.shared(
            0.0001*np.asarray(
                np.random.uniform(
                    low=-4 * np.sqrt(6. / 3+n_hidden_l2),
                    high=4 * np.sqrt(6. / 3+n_hidden_l2),
                    size=(n_visible, 3)
                ), dtype=theano.config.floatX),
            borrow=True
        )

        #self.params = [self.vis_to_l1, self.l1_to_l2, self.l2_to_rvar,
        #               self.l1_biases, self.l2_biases, self.rvar_biases]
        self.params = [self.l2_to_rvar,
                       self.rvar_biases]



    def encoder(self, X):
        #h1 = T.nnet.sigmoid(T.dot(X, self.vis_to_l1) + self.l1_biases)
        #h2 = T.nnet.sigmoid(T.dot(h1, self.l1_to_l2) + self.l2_biases)
        rvar = T.nnet.sigmoid(T.dot(X, self.l2_to_rvar) + self.rvar_biases)
        #rvar = T.dot(X, self.l2_to_rvar) + self.rvar_biases
        return rvar

    def decoder(self, hidden):
        return self.scene(hidden)

    def minimize_cost(self,  X):
        h3 = self.encoder(X)
        reconImage = self.decoder(h3)[:,:,0].flatten()/255.0
        #reconImage = T.sum(self.decoder(h3))
        return T.sum((X-reconImage)**2)
