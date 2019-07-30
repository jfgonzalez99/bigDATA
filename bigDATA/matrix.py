from numpy import array, diag, mean, shape, sign, zeros
from numpy.linalg import det, eig, inv, norm
from numpy.linalg import solve as npsolve
from numpy.random import rand
from scipy.linalg import lu, svd
import matplotlib.pyplot as plt

def evectors(matrix):
    """ Returns the eigenvectors of a given matrix.
    Args
    ---
    `matrix : np.array` A numpy matrix
    
    Returns
    ---
    `vecs : np.array` A numpy matrix containing the eigenvectors
    """
    vals_vecs = eig(matrix)
    return vals_vecs[1]


def evalues(matrix):
    """ Returns the eigenvalues of a given matrix.
    Args
    ---
    `matrix : np.array` A numpy matrix
    
    Returns
    ---
    `vals : np.array` A numpy vector containing the eigenvalues
    """
    vals_vecs = eig(matrix)
    return vals_vecs[0]


def inverse(matrix):
    """ Returns the inverse of a given matrix.
    Args
    ---
    `matrix : np.array` A numpy matrix
    
    Returns
    ---
    `inv : np.array` The inverse matrix
    """
    invMatrix = inv(matrix)
    return invMatrix


def covarianceMatrix(A, B):
    """ Returns the covariance matrix of two given matrices `A` and `B`.
    Args
    ---
    `A : np.array` A `m` x `n` numpy matrix
    `B : np.array` A `m` x `n` numpy matrix
    
    Returns
    ---
    `cMatrix : np.array` The covariance matrix
    """
    N = len(A[0])
    C = (A @ B.T) / N
    return C


def SVDecomp(matrix):
    """ Performs a singlular value decomposition a given matrix.
    Args
    ---
    `matrix : np.array` An `m` x `n` numpy matrix
    
    Returns
    ---
    `U : np.array` An `m` x `m` orthonormal matrix whose columns are the
    eigenvectors of `matrix @ matrix.T`

    `S : np.array` An `m` x `n` matrix with the singular values (square roots
    of the non-zero eigenvalues of `matrix @ matrix.T`) along diagonal

    `V : np.array` An `n` x `n` orthonormal matrix whose columns are the 
    eigenvectors of `matrix.T @ matrix`
    """
    U, S, V = svd(matrix)
    S = diag(S)
    V = V.T
    return U, S, V


def LUDecomp(matrix):
    """ Performs an LU decomposition a given matrix.
    Args
    ---
    `matrix : np.array` A numpy matrix
    
    Returns
    ---
    `P : np.array` The permutation matrix

    `L : np.array` Lower triangular matrix

    `U : np.array` Upper triangular matrix
    """
    P, L, U = lu(matrix)
    P = P.T
    return P, L, U


def polarDecomp(matrix):
    """ Performs a polar decomposition on a given matrix and breaks down the matrix into its rotating and stretching components.
    Args
    ---
    `matrix : np.array` A numpy matrix
    
    Returns
    ---
    `rotate : np.array` The rotation matrix

    `stretch : np.array` The stretch matrix
    """
    U,S,V = SVDecomp(matrix)

    rotate = U@V.T
    stretch = U@S@U.T

    return rotate, stretch


def random(height, width):
    """ Returns a random matrix of a given size.
    Args
    ---
    `height : int` The height of the returned matrix
    
    `width : int` The width of the returned matrix
    
    Returns
    ---
    `randomMatrix : np.array` A random matrix of the desired height and width
    """
    randomMatrix = rand(height, width)
    return randomMatrix


def solve(A, b):
    """ Solve for `x` a system of linear equations in the form of `Ax=b`.
    Args
    ---
    `A : np.array` The left hand matrix
    
    `b : np.array` The right hand vector
    
    Returns
    ---
    `x : np.array` The solution vector
    """
    x = npsolve(A,b)
    return x


def solveMany(A, B):
    """ Solves for many `x`s in a system of linear equations in the form of `Ax=b` where multiple `b`'s are given. Uses LU decomposition to bring time down from `O(n^3/3)` to `O(n^2)`.
    Args
    ---
    `A : np.array` The left hand matrix
    
    `B : np.array` A matrix whose columns are composed of all the right hand vectors
    
    Returns
    ---
    `X : np.array` A matrix whose columns are composed of all the solution vectors that correspond with their respective column in `B`
    """
    P,L,U = LUDecomp(A)
    N = len(B[0])
    X = zeros(shape(B))

    for i in range(N):
        c = solve(L, P @ B[:,i])
        x_i = solve(U, c)
        X[:,i] = x_i

    return X


def perturb(A, b, delta_b):
    """ Perturbs the system `Ax=b` by `delta_b`.
    Args
    ---
    `A : np.array` The left hand matrix
    
    `b : np.array` The right hand vector

    `delta_b : np.array` The perturbing vector
    
    Returns
    ---
    `relError : float` The relative error to the solution caused by `delta_b`
    
    `relPerturbation : float` The relative perturbation of `b` caused by `delta_b`
    """
    x1 = solve(A, b)
    x2 = solve(A, b + delta_b)
    delta_x = x2 - x1

    relError = norm(delta_x)/norm(x1)
    relPerturbation = norm(delta_b)/norm(b)

    return relError, relPerturbation


def optimalFit(X, Y, plot=False):
    """ Given two sets of points, finds the optimal shift and rotation to fit the points in matrix `X` onto `Y`.
    Args
    ---
    `X : np.array` An `m` x `n` matrix that represents the set of points to be shifted and rotated
    
    `Y : np.array` An `m` x `n` matrix that represents the desired set of points to be shifted onto

    `plot : bool` If set to true and data is 2-dimensional will plot the points and ideal transformation
    
    Returns
    ---
    `X_Translation : np.array` An `m` x `1` vector that is the optimal translation of `X` onto `Y`

    `X_Rotation : np.array` An `m` x `m` matrix that is the optimal rotation of `X` onto `Y`

    `X_Translated_Rotated : np.array` `X` after the optimal shift and rotation has been applied
    """
    M = len(X[:,0])
    N = len(X[0])

    # Find center of mass of X and Y
    X_Center = array([mean(X[i]) for i in range(M)])
    Y_Center = array([mean(Y[i]) for i in range(M)])

    # Optimal shift of X
    X_Translation = Y_Center - X_Center

    # Shift X to optimal position
    X_Translated = zeros(shape(X))
    for i in range(N):
        X_Translated[:,i] = X[:,i] + X_Translation

    # Find optimal rotation of X
    C = Y @ X_Translated.T
    SVD = SVDecomp(C)
    U,V = SVD[0], SVD[2]
    X_Rotation = U @ V.T

    # Rotate X to optimal position
    X_Translated_Rotated = zeros(shape(X))
    for i in range(N):
        X_Translated_Rotated[:,i] = X_Rotation @ X_Translated[:,i]

    if plot and (M == 2):
        # Plot original points
        subplt1 = plt.subplot(1, 2, 1)
        hl1, = plt.plot(X[0], X[1], '.', color="red", markersize=7)
        hl2, = plt.plot(Y[0], Y[1], '.', color="blue", markersize=7)
        plt.legend([hl1, hl2], ['X', 'Y'])
        plt.title("Original Points")\

        # Plot tranformed points
        plt.subplot(1, 2, 2, sharex=subplt1, sharey=subplt1)
        hl3, = plt.plot(X_Translated_Rotated[0], X_Translated_Rotated[1], '.', 
                        color="red", markersize=7)
        hl4, = plt.plot(Y[0], Y[1], '.', color="blue", markersize=7)
        plt.legend([hl3, hl4], ['X_Translated_Rotated', 'Y'])
        plt.title("Optimal Transformation")
        plt.show()

    return X_Translation, X_Rotation, X_Translated_Rotated
