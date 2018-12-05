#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt


# Calculate Ground Truth
dt = 0.01
ts = np.arange(0, 1.5 * np.pi, dt)
xs = 2 * (1 - np.cos(ts)) * np.cos(ts)
ys = 2 * (1 - np.cos(ts)) + np.sin(ts)

points = np.stack([xs, ys])

# Location of landmarks
pointA = np.array([0, 0])
pointB = np.array([0, 1])

# Measurements from the marker points
measureA = points - pointA[:, np.newaxis]
measureB = points - pointB[:, np.newaxis]

# Noise standard deviation
noise = 0.1
mean = 0

# Noisy measurements
noisyA = measureA + np.random.normal(mean, noise, measureA.shape)
noisyB = measureA + np.random.normal(mean, noise, measureA.shape)


def FJacobian(x):
    return np.array(
        [[1, 0, - x[3] * np.sin(x[2]), np.cos(x[2]), 0],
         [0, 1, x[3] * np.cos(x[2]), np.sin(x[2]), 0],
         [0, 0, 1, 0, 1],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 0, 1]])


def HJacobian(x):
    return np.array(
        [[1, 0, 0, 0, 0],
         [0, 1, 0, 0, 0]])


def hx(x, landmark):
    return x[:2] + landmark


def kalman(x, P, measurement, R, Q):
    # Predict
    F = FJacobian(x)
    H = HJacobian(x)

    xHat = np.dot(F, x)
    PHat = np.dot(np.dot(F, P), F.T) + Q

    # Update
    y = measurement - hx(xHat, pointA)
    S = np.dot(np.dot(H, P), H.T) + R
    K = np.dot(np.dot(PHat, H.T), np.linalg.inv(S))

    xNew = xHat + np.dot(K, y)
    PNew = np.dot(np.eye(K.shape[0]) - np.dot(K, H), PHat)

    return xNew, PNew


def kalmanSim():
    x = np.array([1, 0, 0, 0, 0])
    P = np.diag([0.1, 0.1, .1, .1, .1])
    R = np.eye(2) * noise
    Q = np.eye(P.shape[0]) / 1e4

    result = []
    for measurement in noisyA.T:
        x, P = kalman(x, P, measurement, R, Q)
        result.append(x[:2])

    return np.array(result).T


simRes = kalmanSim()
ax = plt.gca()
ax.set_aspect('equal')
ax.plot(xs, ys)
ax.scatter(noisyA[0], noisyA[1])
ax.plot(simRes[0], simRes[1])

ax.legend(['Ground Truth', 'Noisy Measurements', 'Predicted'])
ax.figure.savefig('comparison.png')
plt.show()
