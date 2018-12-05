#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt


# Calculate Ground Truth
dt = 0.1
ts = np.arange(0, 2 * np.pi, dt)
xs = np.cos(ts)
ys = np.sin(ts)

points = np.stack([xs, ys])

# Location of landmarks
pointA = np.array([0, 0])

# Measurements from the marker points
measureA = points - pointA[:, np.newaxis]

# Noise standard deviation
noise = 0.1
mean = 0

# Noisy measurements
noisyA = measureA + np.random.normal(mean, noise, measureA.shape)


def kalman(x, P, measurement, R, F, H):
    # Prior mean
    # Calculate residual
    y = measurement - np.dot(H, x)

    # System uncertainty
    S = np.dot(np.dot(H, P), H.T) + R

    # Kalman gain
    K = np.dot(np.dot(P, H.T), np.linalg.inv(S))

    # Prediction
    xHat = x + np.dot(K, y)
    PNew = np.dot(np.eye(K.shape[0]) - np.dot(K, H), P)
    PHat = np.dot(np.dot(F, PNew), F.T)

    return xHat, PHat


def kalmanSim():
    x = np.array([1, 0, 0, 0, 0, 0, 1])
    P = np.diag([0.1, 0.1, 1, 1, 1, 1, 1])
    R = np.eye(2) * noise
    F = np.array(
        [[1, 0, dt, 0],
         [0, 1, 0, dt],
         [0, 0, 1, 0],
         [0, 0, 0, 1]])

    H = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0]])

    result = []
    for measurement in noisyA.T:
        x, P = kalman(x, P, measurement, R, F, H)
        result.append(x[:2])

    return np.array(result).T


simRes = kalmanSim()
ax = plt.gca()
ax.plot(xs, ys)
ax.scatter(noisyA[0], noisyA[1])
ax.plot(simRes[0], simRes[1])

ax.legend(['Ground Truth', 'Noisy Measurements', 'Predicted'])
ax.figure.savefig('comparison.png')
