import numpy as np
import scipy.linalg as linalg

def evectors(matrix):
    """ Returns the eigenvectors of a given matrix.
    Args
    ---
    `matrix : np.array` A numpy matrix
    
    Returns
    ---
    `vecs : np.array` A numpy matrix containing the eigenvectors
    """
    vals, vecs = np.linalg.eig(matrix)
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
    vals, vecs = np.linalg.eig(matrix)
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
    inv = np.linalg.inv(matrix)
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
    U, S, V = linalg.svd(matrix)
    return U, S, V


def LUDecomp(matrix):
    """ Performs an LU decomposition a given matrix.
    Args
    ---
    `matrix : np.array` A numpy matrix
    
    Returns
    ---
    `P : np.array` The permutation matrix

    `L : np.array` Lower diagonal matrix

    `U : np.array` Upper diagonal matrix
    """
    P, L, U = linalg.lu(matrix)
    P = P.T
    return P, L, U
