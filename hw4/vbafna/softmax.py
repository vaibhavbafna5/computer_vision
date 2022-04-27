import pickle
import numpy as np
from layers import *

class SoftmaxClassifier(object):
    """
    A fully-connected neural network with softmax loss that uses a modular
    layer design. We assume an input dimension of D, a hidden dimension
    of H, and perform classification over C classes.

    The architecture should be fc - softmax if no hidden layer.
    The architecture should be fc - relu - fc - softmax if one hidden layer

    Note that this class does not implement gradient descent.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3072, hidden_dim=None, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer, None
          if there's no hidden layer.
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with fc weights and biases using the keys        #
        # 'W' and 'b', i.e., W1, b1 for the weights and bias in the first linear   #
        # layer, W2, b2 for the weights and bias in the second linear layer.       #
        ############################################################################
        self.D = input_dim
        self.M = hidden_dim
        self.C = num_classes
        self.reg = reg

        self.params['W1'] = np.random.normal(scale=weight_scale, size=(input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)

        # self.params['W2'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
        # self.params['b2'] = np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def helper_relu_forward(self, x, w, b):
        a, fc_cache = fc_forward(x, w, b)
        out, relu_cache = relu_forward(a)
        cache = (fc_cache, relu_cache)
        return out, cache


    def helper_relu_backward(self, dout, cache):
        fc_cache, relu_cache = cache
        da = relu_backward(dout, relu_cache)
        dx, dw, db = fc_backward(da, fc_cache)
        return dx, dw, db

    
    def forwards_backwards(self, X, y=None, return_dx = False):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, Din)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass. And
        if return_dx if True, return the gradients of the loss with respect to 
        the input image, otherwise, return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the one-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################

        W1 = self.params['W1']
        b1 = self.params['b1']
        # W2 = self.params['W2']
        # b2 = self.params['b2']

        X = X.reshape(X.shape[0], self.D)

        # PT 2 BELOW 
        out_1, cache_1 = fc_forward(X, W1, b1)
        scores = out_1
        # PT 2 ENDS

        # PT 3 BELOW

        # Forward into first layer
        # hidden_layer, cache_hidden_layer = self.helper_relu_forward(X, W1, b1)
        # # Forward into second layer
        # scores, cache_scores = fc_forward(hidden_layer, W2, b2)

        # PT 3 ENDS

        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the one-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   # 
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        # PT 2 BELOW
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * np.sum(self.params['W1'] ** 2)

        dx_1, grads['W1'], grads['b1'] = fc_backward(dscores, cache_1)
        # PT 2 ENDS

        # PT 3 BELOW

        # data_loss, dscores = softmax_loss(scores, y)
        # reg_loss = 0.5 * self.reg * np.sum(W1**2)
        # reg_loss += 0.5 * self.reg * np.sum(W2**2)
        # loss = data_loss + reg_loss

        # # Backpropagaton
        # grads = {}
        # # Backprop into second layer
        # dx1, dW2, db2 = fc_backward(dscores, cache_scores)
        # dW2 += self.reg * W2

        # # Backprop into first layer
        # dx, dW1, db1 = self.helper_relu_backward(
        #     dx1, cache_hidden_layer)
        # dW1 += self.reg * W1

        # grads.update({'W1': dW1,
        #               'b1': db1,
        #               'W2': dW2,
        #               'b2': db2})

        # return loss, grads

        # PT 3 ENDS

        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def save(self, filepath):
        with open(filepath, "wb") as fp:   
            pickle.dump(self.params, fp, protocol = pickle.HIGHEST_PROTOCOL) 
            
    def load(self, filepath):
        with open(filepath, "rb") as fp:  
            self.params = pickle.load(fp)  
