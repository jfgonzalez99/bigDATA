from numpy import exp as npexp
from numpy import diagflat


def sigmoid(x):
    """ Sigmoid function with input vector x """
    sig = 1 /(1 + npexp(-x))
    return sig


def sigmoid_prime(z):
    """ Derivative of the sigmoid function """
    dsig = sigmoid(z)*(1-sigmoid(z))
    return dsig


def softmax(x):
    """ Softmax function with input vector x """
    exp_x = npexp(x)
    exp_sum = sum(exp_x)
    smax = exp_x / exp_sum
    return smax


def softmax_prime(z):
    """ Derviative of the softmax function """
    smax_prime = diagflat(softmax(z)) - softmax(z) @ softmax(z).T
    return smax_prime


def tansig(x):
    """ Tansig function with input vector x """
    tsig = 2 / (1 + npexp(-2 * x)) - 1
    return tsig


def C(aL, y):
    """ Cost function """
    cost = 0.5*sum((aL-y)**2)
    return cost


def feed_forward(x, W, B, activation, classification):
    """ Feed input forward through network. 
    Args
    ---
    `x : np.array` input vector to be classified
    `W : np.array[]` weights of neural network
    `B : np.array` biases of neural network
    `activation : function` function used on all layers except output
    `classification : function` function used on output

    Returns 
    ---
    `A : np.array[]` values at each step after being passed through activation function
    `Z : np.array[]` values at each step before being passed through activation function
    """
    N = len(B)

    # Add a1 input layer
    A = [x]
    Z = []

    for i in range(N):
        z_i = W[i] @ A[-1] + B[i]
        if i == (N - 1):
            a_i = classification(z_i)
        else:
            a_i = activation(z_i)
        A.append(a_i)
        Z.append(z_i)

    return A, Z


def back_propagation(x, y, W, B, learning_rate):
    """ Performs a single back propagation iteration.
    Args
    ---
    `x : np.array` input vector to be classified
    `y : np.array` true classification labels
    `W : np.array[]` weights of neural network
    `B : np.array` biases of neural network
    `learning_rate : int` the learning rate of the neural network 

    Returns
    ---
    `W : np.array[]` updated weights
    `B : np.array[]` updated biases
    """
    # Feed forward
    A, Z = feed_forward(x, W, B, sigmoid, softmax)

    # Calculate cost and dcost
    # cost = C(A[-1], y)
    dcost = A[-1] - y

    nabla_B = []
    nabla_W = []

    N = len(Z)

    # Back propagation
    for i in range(N):
        step = N - 1 - i
        if i == 0:
           delta_i = softmax_prime(Z[step]) @ dcost
        else:
            delta_i = sigmoid_prime(Z[step]) @ dcost

        nabla_bi = delta_i
        nabla_wi = delta_i @ A[step].T

        nabla_B.insert(0, nabla_bi)
        nabla_W.insert(0, nabla_wi)

    # Adjusts weights and biases
    for i in range(N):
        W[i] -= learning_rate * nabla_W[i]
        B[i] -= learning_rate * nabla_B[i]

    return W, B
