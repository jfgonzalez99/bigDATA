from difflib import ndiff
from numpy import arccos

def editDistance(str1, str2):
    """ Returns the number of edits needed to turn string `str1` into string `str2`.
    Args
    ---
    `str1 : string` The first string

    `str2 : string` The second string

    Returns
    ---
    `distance : int` The number of edits needed
    """
    # Create an array of the changes needed to change s1 -> s2
    changes = [change for change in ndiff(str1, str2) if change[0] != ' ']
    distance = len(changes)
    return distance


def jaccardDistance(a, b):
    """ Returns the percentage of elements in set `a` or `b` that are not in both `a` and `b`.
    Args
    ---
    `a : set` The first set

    `b : set` The second set

    Returns
    ---
    `jDistance : float` The percent of elements not in both `a` and `b`
    """
    a = set(a)
    b = set(b)
    # Calculate Jaccard similarity
    jSimilarity = len(a & b) / len(a | b)
    jDistance =  1 - jSimilarity
    return jDistance


def hammingDistance(str1, str2):
    """ Returns the number of `i`th characters in `str1` that don't match the `i`th character in `str2`.
    Args
    ---
    `str1 : string` The first string

    `str2 : string` The second string

    Returns
    ---
    `differences : int` The differences between `str1` and `str2`
    """
    # Convert strings to arrays
    a = list(str1)
    b = list(str2)
    # Determine what n equals
    if len(a) < len(b):
        n = len(a)
    else:
        n = len(b)
    # Increment the number of distances for each difference
    differences = 0
    for i in range(n):
        if a[i] != b[i]:
            differences += 1
    return differences


def cosineDistance(a, b):
    """ Calculates the cosine distance between lists of numbers `a` and `b`.
    Args
    ---
    `a : float[]` The first list of floats (or ints)

    `b : float[]` The second list of floats (or ints)

    Returns
    ---
    `distance : float` The distance between `a` and `b`
    """
    n = len(a)
    numerator = 0
    denominatorA = 0
    denominatorB = 0
    for i in range(n):
        numerator += a[i] * b[i]
        denominatorA += a[i]**2
        denominatorB += b[i]**2
    num = numerator / (denominatorA*denominatorB)**(1/2)
    distance = arccos(num)
    return distance


# the Euclidean (Lr) distance between a and b
def lrDistance(a, b, r):
    """ Calculates the distance between `a` and `b` in `r` dimensional space.
    Args
    ---
    `a : float[]` The first set

    `b : float[]` The second set

    `r : int` The dimension of space to find the distance in (the string "infinity" can also be passed)

    Returns
    ---
    `distance : float` The distance between `a` and `b`
    """
    n = len(a)
    if (r == "infinity"):
        differences = []
        for i in range(n):
            differences.append(abs(a[i]-b[i]))
        return max(differences)
    else:
        sum = 0
        for i in range(n):
            sum += abs(a[i] - b[i])**r
        return sum**(1/r)
