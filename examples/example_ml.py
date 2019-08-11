import bigDATA.ml as ml
import numpy as np

# Classify Xinput using given weights and biases
x = np.array([[8], [7]])

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

W = [w2, w3, w4]
B = [b2, b3, b4]

cNetwork = ml.Classification(W, B, ml.sigmoid)

# y is the true classification of x
y = cNetwork.feed_forward(x)

print("True classification:")
print(y)

# Perturb some of the weights and biases to see effect of back propagation
W_perturbed = W
B_perturbed = B
perturbation = 10

for i in range(2):
    W_perturbed[i] = W[i] + perturbation
    B_perturbed[i] = B[i] + perturbation

# Create new network with perturbed weights and biases
perturbedNetwork = ml.Classification(W_perturbed, B_perturbed, ml.sigmoid)

# Feed perturbed weights and biases forward
classification_0 = perturbedNetwork.feed_forward(x)

# Perform one back propagation iteration
perturbedNetwork.back_propagation(x, y, 2500)

# Feed forward and update values for A and Z
classification_1 = perturbedNetwork.feed_forward(x)

# Compare values before and after back propagation
print("Prior classification:")
print(classification_0)
print("New classification:")
print(classification_1)

print("Prior cost:")
print(ml.cost(classification_0, y))
print("New cost:")
print(ml.cost(classification_1, y))
