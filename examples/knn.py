import bigDATA.ml as ml
import numpy as np
import matplotlib.pyplot as plt

#====== Discrete classification ======#
# Training data
X = np.array([[3, 3, 3, 8, 5, 2, 4, 6,  3, 8, 2, 0, 4, 1, 8, 9, 9, 8, 7, 2, 2,
               9, 8, 7, 1, 2, 7, 7, 4, 3, 2, 6, 9],
              [7, 5, 1, 5, 3, 9, 5, 7, 10, 7, 8, 2, 3, 1, 6, 7, 8, 8, 3, 8, 8,
               9, 4, 9, 4, 7, 5, 0, 8, 6, 10, 2, 9]])

# Training labels
y = np.array([3, 3, 3, 1, 1, 2, 3, 2, 2, 2, 3, 3, 3, 3, 2, 2,
              2, 2, 1, 3, 3, 2, 1, 2, 3, 3, 1, 1, 2, 3, 2, 1, 2])

N = len(y)

# Red is group 1, Green is group 2, Blue is group 3
color = ["r", "g", "b"]
# Plot the data
for i in range(N):
    plt.plot(X[:, i][0], X[:, i][1], color=color[y[i] - 1], marker=".")
plt.plot(3.5, 7.5, color="m", marker="x")
plt.title("data")
plt.show()

# Point to be classified
point = np.array([3.5, 7.5])

kNN = ml.KNN(X, y)

for k in [2, 3, 4, 5]:
    classification = kNN.classify(point, k)
    print("k = " + str(k) + ":")
    print(classification)

#====== Regression ======#
X = np.array([[-0.87, -0.52, -0.73, -0.3, 0.75, 0.87, 0.03, 0.4, -0.54, -0.18, -0.69, 0.57, 0.62, -0.02, 0.01, 0.02, 0.12, 0.91, 0.37, -0.46],
              [-0.26, 0.48, 0.23, 0.47, -0.73, 0.08, -0, 0.19, 0.84, -0.61, 0.77, 0.1, -0.68, 0.58, -0.66, -0.24, -0.25, -0.57, -0.99, 0.24]])
y = np.array([1.02, -0.21, 0.3, -0.38, 1.29, 0.68, 0, -0.03, -0.55,
              0.64, -0.29, 0.22, 1.06, -0.58, 0.66, 0.24, 0.26, 1.4, 1.13, -0.03])

N = len(y)

for i in range(N):
    plt.plot(X[:, i][0], X[:, i][1], marker=".", color="b")
plt.plot(0.7, -0.7, color="r", marker="x")
plt.title("data")
plt.show()

point = np.array([0.7, -0.7])

kReg = ml.KNN(X, y)
classification = kReg.kRegression(point, 4, 0.3)

print("k = 4 regression:")
print(classification)
