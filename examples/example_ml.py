import bigDATA.ml as ml
import numpy as np

# Classify Xinput using given weights and biases
xInput = np.array([[8], [7]])

w2 = np.array([[-0.147961,  2.73431],
               [1.99734,    0.91873]])
b2 = np.array([[-13.5989],
               [-12.719]])
w3 = np.array([[-10.6819, -13.7296],
               [-9.49413,  11.1605]])
b3 = np.array([[8.28166],
               [-6.05327]])
w4 = np.array([[18.0988,   -1.46339],
               [-6.17681,  15.0929],
               [-10.6842, -14.0763]])
b4 = np.array([[-4.93757],
               [-4.6388],
               [9.94525]])

W = [w2,w3,w4]
B = [b2,b3,b4]

A, Z = ml.feed_forward(xInput, W, B, ml.sigmoid, ml.softmax)

for i in range(len(A)):
    print("a_" + str(i) + ": ")
    print(A[i])

for i in range(len(Z)):
    print("z_" + str(i + 1) + ": ")
    print(Z[i])

# Save true classification for later
y = A[-1]

# Perturb some of the weights and biases to see effect of back propagation
W_perturbed = W
B_perturbed = B
perturbation = 10

for i in range(2):
    W_perturbed[i] = W[i] + perturbation
    B_perturbed[i] = B[i] + perturbation

# Feed perturbed weights and biases forward
A_perturbed, Z_perturbed = ml.feed_forward(
    xInput, W_perturbed, B_perturbed, ml.sigmoid, ml.softmax)

# Perform one back propagation iteration
W_updated, B_updated = ml.back_propagation(
    xInput, y, W_perturbed, B_perturbed, 2500)

# Feed forward and update values for A and Z
A_updated, Z_updated = ml.feed_forward(
    xInput, W_updated, B_updated, ml.sigmoid, ml.softmax)

# Compare values before and after back propagation
print("Prior cost:")
print(ml.C(A_perturbed[-1], y))
print("New cost:")
print(ml.C(A_updated[-1], y))
print("Prior z4:")
print(Z_perturbed[-1])
print("New z4:")
print(Z_updated[-1])
print("Prior a4:")
print(A_perturbed[-1])
print("New a4:")
print(A_updated[-1])
