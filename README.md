# bigDATA

A python library for big data analysis. This package is a collection of functions that I have found to be useful in my coursework on big data and machine learning.

## Requirements and Installation

This package requires that **numpy**, **scipy**, and **matplotlib** be installed. Using pip this can be done with the following command:

```pip install numpy scipy matplotlib```

or see [Scipy's website](https://www.scipy.org/install.html) for more information on installation.

To install the **bigDATA** package using pip simply run:

```pip install -i https://test.pypi.org/simple/ bigDATA```

## bigDATA.matrix

* **`random`**: returns a random matrix of a certain size
* **`evalues`**: returns eigenvalues of a matrix
* **`evectors`**: returns eigenvectors of a matrix
* **`evectors`**: returns inverse of a matrix
* **`evectors`**: returns the covariance matrix of a matrix
* **`SVDecomp`**: performs a singular value decomposition
* **`LUDecomp`**: performs an LU decomposition
* **`polarDecomp`**: performs a polar decomposition on a given matrix and breaks down the matrix into its rotating and stretching components
* **`solve`**: solves a linear system of equations
* **`solveMany`**: Solves for many `x`s in a system of linear equations in the form of `Ax=b` where multiple `b`'s are given
* **`perturb`**: perturbs a system of equations and returns the relative perturbation and error
* **`optimalFit`**: given two sets of points, finds the optimal shift and rotation to fit the points in matrix `X` onto `Y`


## bigDATA.similarity

* **`editDistance`**: returns the number of edits needed to turn string `str1` into string `str2`
* **`jaccardDistance`**: returns the percentage of elements in set `a` or `b` that are not in both `a` and `b`
* **`hammingDistance`**: Returns the number of `i`th characters in `str1` that don't match the `i`th character in `str2`
* **`cosineDistance`**: calculates the cosine distance between lists of numbers `a` and `b`
* **`lrDistance`**: calculates the L_r distance between `a` and `b` in `r` dimensional space


## bigDATA.ml

* **`sigmoid`**, **`sigmoid_prime`**, **`softmax`**, **`softmax_prime`**, and **`tansig`** activation functions
* **`cost`**: cost function
* **`Classification`**:
    * **`feed_forward`**: feed input forward through a neural network
    * **`back_propagation`**: performs a single back propagation iteration
