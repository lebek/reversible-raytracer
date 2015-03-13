# Reversible Ray Tracer
The Reversible Ray Tracer (RRT) is a program that takes a description of a 3D scene and outputs a picture of the scene. But it doesn’t end there. Say what you want to change about the output and RRT can backtrack to the inputs and update them to bring about the intended change.

## Dependencies
* numpy
* scipy
* theano
* tornado

## Usage
```
$ python server.py
```

Now open `index.html` in your browser. Click a point to maximize the brightness at that point.

## How it works
![Maximizing the brightness at a point](https://raw.githubusercontent.com/lebek/RRT/master/demo-gifs/2.gif)

The optimization works by gradient descent (or ascent). Here’s the algorithm:

1. Define a loss function on the output.

2. Move through the input parameter space in proportion to the gradient of the loss function w.r.t. each input parameter. At each iteration:

  1. Calculate the partial derivative of the loss w.r.t. each input parameter.
  2. Update each input parameter proportionally to the gradient.

3. Continue until the loss stops improving much.

Todo:
* Shadows
* More interesting loss functions

