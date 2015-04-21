# Reversible Raytracer
Reversible Raytracer is a program that takes a description of a 3D scene and outputs a picture of the scene. But it doesn’t end there. Say what you want to change about the output and it can backtrack to the inputs and update them to bring about the intended change.

## Dependencies
* numpy
* scipy
* theano (bleeding-edge, or at least including [this commit](https://github.com/Theano/Theano/commit/ba5f2a3abd40377819d652ebf7d43b151c64ea87))

## Usage
```
$ THEANO_FLAGS='mode=FAST_RUN,floatX=float32' python optimize_brightness.py
or
$ THEANO_FLAGS='mode=FAST_RUN,floatX=float32' python match_mirror.py
```

The result is in `output/`

## How it works
The optimization works by gradient descent (or ascent). Here’s the algorithm:

1. Define a loss function on the output.

2. Move through the input parameter space in proportion to the gradient of the loss function w.r.t. each input parameter. At each iteration:

  1. Calculate the partial derivative of the loss w.r.t. each input parameter.
  2. Update each input parameter proportionally to the gradient.

3. Continue until the loss stops improving much.
