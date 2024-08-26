import numpy as np

from nndl.layers import *
from nndl.conv_layers import *
from utils.fast_layers import *
from nndl.layer_utils import *
from nndl.conv_layer_utils import *

import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize the weights and biases of a three layer CNN. To initialize:
    #     - the biases should be initialized to zeros.
    #     - the weights should be initialized to a matrix with entries
    #         drawn from a Gaussian distribution with zero mean and 
    #         standard deviation given by weight_scale.
    # ================================================================ #

    C, H, W = input_dim
    pad = (filter_size - 1) / 2

    self.params["W1"] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
    self.params["b1"] = np.zeros(num_filters)

    conv_height = H - filter_size + 2 * pad + 1
    conv_width = W - filter_size + 2 * pad + 1
    pool_height = int((conv_height - 2) / 2 + 1)
    pool_width = int((conv_width - 2) / 2 + 1)

    input_size = pool_height * pool_width * num_filters
    self.params["W2"] = weight_scale * np.random.randn(input_size, hidden_dim)
    self.params["b2"] = np.zeros(hidden_dim)

    self.params["W3"] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params["b3"] = np.zeros(num_classes)

    self.bn_params = {}
    if use_batchnorm:
      self.bn_params["bn_param1"] = {"mode": "train"}
      self.bn_params["bn_param2"] = {"mode": "train"}
      self.params["gamma1"] = np.ones(num_filters)
      self.params["beta1"] = np.zeros(num_filters)
      self.params["gamma2"] = np.ones(hidden_dim)
      self.params["beta2"] = np.zeros(hidden_dim)

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of the three layer CNN.  Store the output
    #   scores as the variable "scores".
    # ================================================================ #

    if self.use_batchnorm:
      gamma1, beta1, bn_param1 = self.params["gamma1"], self.params["beta1"], self.bn_params["bn_param1"]
      conv_out, conv_cache = conv_forward_fast(X, W1, b1, conv_param)
      bn1_out, bn1_cache = spatial_batchnorm_forward(conv_out, gamma1, beta1, bn_param1)
      relu1_out, relu1_cache = relu_forward(bn1_out)
      pool_out, pool_cache = max_pool_forward_fast(relu1_out, pool_param)

      gamma2, beta2, bn_param2 = self.params["gamma2"], self.params["beta2"], self.bn_params["bn_param2"]
      affine_out, affine_cache = affine_forward(pool_out, W2, b2)
      bn2_out, bn2_cache = batchnorm_forward(affine_out, gamma2, beta2, bn_param2)
      relu2_out, relu2_cache = relu_forward(bn2_out)
      scores, scores_cache = affine_forward(relu2_out, W3, b3)

    else:
      conv_out, conv_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
      affine_out, affine_cache = affine_relu_forward(conv_out, W2, b2)
      scores, scores_cache = affine_forward(affine_out, W3, b3)

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    if y is None:
      return scores
    
    loss, grads = 0, {}
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass of the three layer CNN.  Store the grads
    #   in the grads dictionary, exactly as before (i.e., the gradient of 
    #   self.params[k] will be grads[k]).  Store the loss as "loss", and
    #   don't forget to add regularization on ALL weight matrices.
    # ================================================================ #

    loss, grad_score = softmax_loss(scores, y)
    norms = np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))
    loss += 0.5 * self.reg * norms

    grad_Z2, grads["W3"], grads["b3"] = affine_backward(grad_score, scores_cache)

    if self.use_batchnorm:
      grad_relu2 = relu_backward(grad_Z2, relu2_cache)
      grad_B2, grads["gamma2"], grads["beta2"] = batchnorm_backward(grad_relu2, bn2_cache)
      grad_Z1, grads["W2"], grads["b2"] = affine_backward(grad_B2, affine_cache)

      grad_pool = max_pool_backward_fast(grad_Z1, pool_cache)
      grad_relu1 = relu_backward(grad_pool, relu1_cache)
      grad_B1, grads["gamma1"], grads["beta1"] = spatial_batchnorm_backward(grad_relu1, bn1_cache)
      grad_X, grads["W1"], grads["b1"] = conv_backward_fast(grad_B1, conv_cache)

    else:
      grad_Z1, grads["W2"], grads["b2"] = affine_relu_backward(grad_Z2, affine_cache)
      grad_X, grads["W1"], grads["b1"] = conv_relu_pool_backward(grad_Z1, conv_cache)

    grads["W3"] += self.reg * W3
    grads["W2"] += self.reg * W2
    grads["W1"] += self.reg * W1

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return loss, grads
