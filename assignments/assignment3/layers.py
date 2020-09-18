import numpy as np
import math

def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    loss = reg_strength * np.trace(np.matmul(W.T, W))   # L2(W) = λ * tr(W.T * W)
    grad = 2 * reg_strength * W                         # dL2(W)/dW = 2 * λ * W
    
    return loss, grad


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    single = (predictions.ndim == 1)

    if single:
        predictions = predictions.reshape(1, predictions.shape[0])

    maximums = np.amax(predictions, axis=1).reshape(predictions.shape[0], 1)
    predictions_ts = predictions - maximums

    predictions_exp = np.exp(predictions_ts)
    sums = np.sum(predictions_exp, axis=1).reshape(predictions_exp.shape[0], 1)
    result = predictions_exp / sums

    if single:
        result = result.reshape(result.size)

    return result    


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    new_loss = np.vectorize(lambda x: -math.log(x))
    if len(probs.shape) == 1:
        probs_target = probs[target_index]
        size_target = 1
    else:
        batch_size = np.arange(target_index.shape[0])
        probs_target = probs[batch_size,target_index.flatten()]
        size_target = target_index.shape[0]
    loss = np.sum(new_loss(probs_target)) / size_target
    return loss

def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO copy from the previous assignment
    prediction = predictions.copy()
    probs = softmax(prediction) 
    loss = cross_entropy_loss(probs, target_index)
    d_preds = probs
    if len(predictions.shape)==1:
        d_preds[target_index] -= 1
    else:
        batch_size = np.arange(target_index.shape[0])
        d_preds[batch_size,target_index.flatten()] -= 1
        d_preds = d_preds/target_index.shape[0]

    return loss, d_preds


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO copy from the previous assignment
        self.d_out_result = np.greater(X, 0).astype(float)  # dZ/dX
        return np.maximum(X, 0)                             # Z

    def backward(self, d_out):
        # TODO copy from the previous assignment
        d_result = np.multiply(d_out, self.d_out_result)    # dL/dX = dL/dZ * dZ/dX
        return d_result     # dL/dX

    def params(self):
        return {}
    def reset_grad(self):
        pass


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None
        self.dw = None

    def forward(self, X):
        # TODO copy from the previous assignment
        self.X = X
        return np.matmul(X, self.W.value) + self.B.value

    def backward(self, d_out):
        # TODO copy from the previous assignment
        
        d_input = np.matmul(d_out, self.W.value.T)     # dL/dX = dL/dZ * dZ/dX = dL/dZ * W.T
        dLdW = np.matmul(self.X.T, d_out)               # dL/dW = dL/dZ * dZ/dW = X.T * dL/dZ
        dLdB = 2 * np.mean(d_out, axis=0)               # dL/dB = dL/dZ * dZ/dB = I * dL/dZ

        self.W.grad += dLdW
        self.B.grad += dLdB

        return d_input     # dL/dX

    def params(self):
        return { 'W': self.W, 'B': self.B }
    def reset_grad(self):
        self.W.grad = np.zeros_like(self.W.value)
        self.B.grad = np.zeros_like(self.B.value)

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape

        self.X = X

        if self.padding:
            self.X = np.zeros((batch_size,
                               height + 2 * self.padding,
                               width + 2 * self.padding,
                               channels), dtype=X.dtype)
            self.X[:, self.padding: -self.padding, self.padding: -self.padding, :] = X

        _, height, width, channels = self.X.shape

        out_height = height - self.filter_size + 1
        out_width = width - self.filter_size + 1

        output = []

        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            row = []
            for x in range(out_width):
                x_filter = self.X[:, y: y + self.filter_size, x: x + self.filter_size, :]
                x_filter = np.transpose(x_filter, axes=[0, 3, 2, 1]).reshape((batch_size, self.filter_size * self.filter_size * channels))
                
                W_filter = np.transpose(self.W.value, axes=[2, 0, 1, 3])
                out = x_filter.dot(W_filter.reshape((self.filter_size * self.filter_size * self.in_channels, self.out_channels)))
                
                # out has shape (batch_size, out_channel)
                row.append(np.array([out], dtype=self.W.value.dtype).reshape((batch_size, 1, 1, self.out_channels)))
            output.append(np.dstack(row))
        output = np.hstack(output)
        output += self.B.value

        return output


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output
        d_in = np.zeros(self.X.shape)
        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                d_filter = d_out[:, y, x, :]

                X_filter = self.X[:, y: y + self.filter_size, x: x + self.filter_size, :]

                X_filter = np.transpose(X_filter, axes=[0, 3, 1, 2])
                X_filter = X_filter.reshape((batch_size, self.filter_size * self.filter_size * channels))
                X_transpose = X_filter.transpose()

                W_filter = np.transpose(self.W.value, axes=[2, 0, 1, 3])
                W_filter = W_filter.reshape((self.filter_size * self.filter_size * self.in_channels, self.out_channels))
                W_transpose = W_filter.transpose()

                d_W_filter = np.dot(X_transpose, d_filter)
                d_W_filter = d_W_filter.reshape(self.in_channels, self.filter_size, self.filter_size, self.out_channels)
                d_W_transpose = np.transpose(d_W_filter, axes=[2, 1, 0, 3])

                self.W.grad += d_W_transpose
                E = np.ones(shape=(1, batch_size))
                B = np.dot(E, d_filter)
                B = B.reshape((d_filter.shape[1]))
                self.B.grad += B

                d_in_xy = np.dot(d_filter, W_transpose)#d_filter.dot(w_filter.transpose())
                d_in_xy = d_in_xy.reshape((batch_size, channels, self.filter_size, self.filter_size))

                d_in_xy = np.transpose(d_in_xy, axes=[0, 3, 2, 1])

                d_in[:, y: y + self.filter_size, x: x + self.filter_size, :] += d_in_xy

        if self.padding:
            d_in = d_in[:, self.padding: -self.padding, self.padding: -self.padding, :]

        return d_in


    def params(self):
        return { 'W': self.W, 'B': self.B }
    def reset_grad(self):
        self.W.grad = np.zeros_like(self.W.value)
        self.B.grad = np.zeros_like(self.B.value)


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension

        self.X = X

        output = []
        for y in range(0, height, self.stride):
            row = []
            for x in range(0, width, self.stride):
                X_filter = X[:, y: y + self.pool_size, x: x + self.pool_size, :]
                row.append(X_filter.max(axis=1).max(axis=1).reshape(batch_size, 1, 1, channels))
            row = np.dstack(row)
            output.append(row)
        output = np.hstack(output)

        return output

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape

        output = np.zeros(self.X.shape)
        
        for y_num, y in enumerate(range(0, height, self.stride)):
            for x_num, x in enumerate(range(0, width, self.stride)):
                d_filter = d_out[:, y_num, x_num, :]
                d_filter = d_filter.reshape(batch_size, 1, 1, channels)
                X_filter = self.X[:, y: y + self.pool_size, x: x + self.pool_size, :]
                d_filter_out = (X_filter == X_filter.max(axis=1).max(axis=1).reshape(batch_size, 1, 1, channels)) * d_filter
                output[:, y: y + self.pool_size, x: x + self.pool_size, :] += d_filter_out.astype(np.float32)
        return output

    def params(self):
        return {}
    def reset_grad(self):
        pass


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        self.X_shape = X.shape
        return X.reshape((batch_size, height * width * channels))

    def backward(self, d_out):
        # TODO: Implement backward pass
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
    def reset_grad(self):
        pass
