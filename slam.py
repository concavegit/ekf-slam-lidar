# 1/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt


dt = 0.033


def FJacobian(x):
    return np.array(
        [[1, 0, - x[3] * np.sin(x[2]) * dt, np.cos(x[2]) * dt,
          0, 0, 0, 0, 0, 0, 0],
         [0, 1, x[3] * np.cos(x[2]) * dt, np.sin(x[2])
          * dt, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, dt, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])


def HJacobian(x):
    return np.array(
        [[-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [-1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [-1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1]])


def hx(x):
    return x[-6:]


def kalman(x, P, measurement, R, Q):
    F = FJacobian(x)
    H = HJacobian(x)

    xHat = np.dot(F, x)
    PHat = np.dot(np.dot(F, P), F.T) + Q

    y = measurement - np.dot(HJacobian(xHat), xHat)
    S = np.dot(np.dot(H, PHat), H.T) + R
    K = np.dot(np.dot(PHat, H.T), np.linalg.inv(S))

    xNew = xHat + np.dot(K, y)
    PNew = np.dot(np.eye(K.shape[0]) - np.dot(K, H), PHat)

    return xNew, PNew


# Calculate Ground Truth
dt = 0.033
ts = np.arange(0, 1.5 * np.pi, dt)
# xs = ts * 0
# ys = ts * 0
xs = 2 * (1 - np.cos(ts)) * np.cos(ts)
ys = 2 * (1 - np.cos(ts)) + np.sin(ts)

points = np.stack([xs, ys])

# Location of landmarks
pointA = np.array([0, 0])
pointB = np.array([0, 1])
pointC = np.array([1, -2])
# beacons = np.stack([pointA, pointB, pointC])
noise = 0.1
beaconNoise = 2
mean = 0
beacons = np.random.normal(mean, beaconNoise, (3, 2))

# Measurements from the marker points
measureA = beacons[0, :, np.newaxis] - points
measureB = beacons[1, :, np.newaxis] - points
measureC = beacons[2, :, np.newaxis] - points

# Noise standard deviation

# Noisy measurements
noisyA = measureA + np.random.normal(mean, noise, measureA.shape)
noisyB = measureB + np.random.normal(mean, noise, measureB.shape)
noisyC = measureC + np.random.normal(mean, noise, measureC.shape)

measurements = np.vstack([noisyA, noisyB, noisyC]).T


def kalmanSim():
    x = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    P = np.diag([0, 0, (np.pi / 3) ** 2, 1, 1 / 9, beaconNoise, beaconNoise,
                 beaconNoise, beaconNoise, beaconNoise, beaconNoise])
    R = np.eye(6) * noise
    # Q = np.eye(P.shape[0]) / 1e3
    Q = np.diag([.03, .03, .03, .03, .03, .001, .001, .001, .001, .001, .001])

    result = []
    for measurement in measurements:
        print(measurement)
        x, P = kalman(x, P, measurement, R, Q)
        result.append(x)

    return np.array(result).T


simRes = kalmanSim()

ax = plt.gca()
ax.set_aspect('equal')
ax.plot(xs, ys)
ax.plot(simRes[0], simRes[1])
ax.plot(simRes[5], simRes[6])
ax.plot(simRes[7], simRes[8])
ax.plot(simRes[9], simRes[10])
ax.scatter(beacons[:, 0], beacons[:, 1], marker=(5, 1), s=100)

ax.legend(['Ground Truth', 'Predicted Odometry', 'Predicted Beacon 1',
           'Predicted Beacon 2', 'Predicted Beacon 3', 'Actual Beacons'])
# ax.figure.savefig('comparison.png')
plt.show()