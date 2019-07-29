from numpy.linalg import eig as eig
from numpy.linalg import inv as inverse
from scipy.linalg import lu as lu
from scipy.linalg import svd as svd


def evectors(matrix):
    """ Returns the eigenvectors of a given matrix.
    Args
    ---
    `matrix : np.array` A numpy matrix
    
    Returns
    ---
    `vecs : np.array` A numpy matrix containing the eigenvectors
    """
    vals, vecs = eig(matrix)
    return vecs


def evalues(matrix):
    """ Returns the eigenvalues of a given matrix.
    Args
    ---
    `matrix : np.array` A numpy matrix
    
    Returns
    ---
    `vals : np.array` A numpy vector containing the eigenvalues
    """
    vals, vecs = eig(matrix)
    return vals


def inverse(matrix):
    """ Returns the inverse of a given matrix.
    Args
    ---
    `matrix : np.array` A numpy matrix
    
    Returns
    ---
    `inv : np.array` The inverse matrix
    """
    inv = inverse(matrix)
    return inv


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
