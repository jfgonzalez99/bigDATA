from numpy.linalg import eig, inv, norm
from numpy.linalg import solve as npsolve
from numpy.random import rand
from numpy import diag, shape, zeros
from scipy.linalg import lu, svd


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


def covarianceMatrix(matrix):
    """ Returns the covariance matrix of a given matrix.
    Args
    ---
    `matrix : np.array` A numpy matrix
    
    Returns
    ---
    `cMatrix : np.array` The covariance matrix
    """
    N = len(matrix[0])
    cMatrix = (matrix @ matrix.T) / N
    return cMatrix


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


def perturb(A,b,delta_b):
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
