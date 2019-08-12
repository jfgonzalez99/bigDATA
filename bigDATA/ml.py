from numpy import exp as npexp
from numpy import diagflat, array, argpartition, zeros
from similarity import lrDistance
from collections import Counter
from matrix import inverse
from operator import itemgetter


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


def linear(x):
    """ Linear function with input vector x """
    return x


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


# class Regression:
#     def __init__(self, W, B, activation):
#         """ Setup your regression network
#         Args
#         ---
#         `W : np.array[]` weights of neural network

#         `B : np.array` biases of neural network

#         `activation : function` function used on all layers except output
#         """
#         self.W = W
#         self.B = B
#         self.activation = activation
#         self.regression = regression

#     def contour(self, X, y):
#         """ Calculate the contour
#         Args
#         ---
#         `X : np.array` a matrix containing the training data points

#         `y : np.array` the classification of the points

#         `activation : function` function used on all layers except output
#         """
#         N = len(y)
#         classifications = np.zeros((N,N))

#         for i in range(N):
#             for j in range(N):
#                 X


class KNN:
    def __init__(self, X, y):
        """ Set up k nearest neighbors classification
        Args
        ---
        `T : np.array` training data

        `y : np.array` classification labels of training data
        """
        self.X = X
        self.y = y
        self.dim = len(X[:, 0])
        self.N = len(y)

    def classify(self, point, k):
        """ Classifies a point using k nearest neighbors
        Args
        ---
        `point : np.array` the point to be classified

        `k : int` the number of nearest neighbors

        Returns
        ---
        `classification : int` the classification label of the given point
        """
        distances = []
        for i in range(self.N):
            distance = lrDistance(point, self.X[:, i], self.dim)
            distances.append(distance)

        # Find indices of nearest neighbors
        ind_neighbors = argpartition(distances, k)[:k]

        # Find labels of k nearest neighbors
        neighbors = []
        for i in ind_neighbors:
            neighbors.append(self.y[i])

        # Find most common label in nearest neighbors
        c = Counter(neighbors)
        classification = c.most_common()[0][0]

        return classification

    def kRegression(self, point, k, weight):
        """ Performs a locally weighted kernel regression
        Args
        ---
        `point : np.array` the point to be classified

        `k : int` the number of nearest neighbors

        `weight : float` how strongly the closeness of the neighbor is taken into consideration

        Returns
        ---
        `classification : float` the classification label of the given point
        """
        distances = []
        weights = []
        for i in range(self.N):
            distance = lrDistance(point, self.X[:, i], self.dim)
            distances.append(distance)
            weights.append(npexp(-distance/weight))

        # Find indices of nearest neighbors
        ind_neighbors = argpartition(distances, k)[:k]

        # Sort neighbors by weight
        neigbor_weights = []
        for i in ind_neighbors:
            neigbor_weights.append((i, weights[i]))
        nw = reversed(sorted(neigbor_weights, key=itemgetter(1)))

        # Matrix of nearest points
        P = zeros((self.dim, k))
        # Diagonal matrix of weights
        K = zeros((k, k))
        # Labels vector
        f = zeros(k)

        column = 0
        for i, weight in nw:
            P[:, column] = self.X[:, i]
            K[column][column] = weight
            f[column] = self.y[i]
            column += 1

        w = inverse(P@K@P.T)@(P@K)@f
        classification = w.T @ point

        return classification
