from numpy import exp as npexp
from numpy import diagflat, array


def sigmoid(x):
    """ Sigmoid function with input vector x """
    sig = 1 / (1 + npexp(-x))
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


def cost(aL, y):
    """ Cost function """
    c = 0.5*sum((aL-y)**2)
    return c


class Classification:
    def __init__(self, W, B, activation):
        """ Setup your classification network
        Args
        ---
        `W : np.array[]` weights of neural network

        `B : np.array` biases of neural network

        `activation : function` function used on all layers except output
        """
        self.W = W
        self.B = B
        self.activation = activation
        if activation.__name__ == "sigmoid":
            self.activation_prime = sigmoid_prime
        else:
            print("WARNING: This activation function is not yet fully supported.")

    def feed_forward(self, x):
        """ Feed input forward through network. 
        Args
        ---
        `x : np.array` input vector to be classified

        Returns 
        ---
        `classification : np.array` the classification of the input vector
        """
        N = len(self.B)

        # A is the values at each step after being passed through activation function
        # Add a1 input layer
        A = [x]
        # Z is the values at each step before being passed through activation function
        Z = []

        for i in range(N):
            z_i = self.W[i] @ A[-1] + self.B[i]
            if i == (N - 1):
                a_i = softmax(z_i)
            else:
                a_i = self.activation(z_i)
            A.append(a_i)
            Z.append(z_i)

        # Store for user's use
        self.A = A
        self.Z = Z

        classification = A[-1]
        return classification

    def back_propagation(self, x, y, learning_rate):
        """ Performs a single back propagation iteration.
        Args
        ---
        `x : np.array` input vector to be classified

        `y : np.array` true classification labels

        `learning_rate : int` the learning rate of the neural network
        """
        # Feed forward if not already done
        classification = self.feed_forward(x)

        # Calculate dcost
        dcost = classification - y

        nabla_B = []
        nabla_W = []

        N = len(self.Z)

        # Back propagation
        for i in range(N):
            step = N - 1 - i
            if i == 0:
                delta_i = softmax_prime(self.Z[step])@dcost
            else:
                delta_i = (self.W[step + 1].T@nabla_B[0]) * \
                    self.activation_prime(self.Z[step])

            nabla_bi = delta_i
            nabla_wi = delta_i @ self.A[step].T

            nabla_B.insert(0, nabla_bi)
            nabla_W.insert(0, nabla_wi)

        # Adjusts weights and biases
        for i in range(N):
            self.W[i] -= learning_rate * nabla_W[i]
            self.B[i] -= learning_rate * nabla_B[i]
