# 1/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

dt = 0.033

def FJacobian(x):
    '''
    Calculate the jacobian of the prediction model at x.

    @param x the 11 dimensional state vector about which to approximate.
    @return the linear estimation of the prediction model about x.
    '''
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


def fx(x):
    '''
    Predict the next state.
    @param x the 11 dimensional state vector from which to predict the
    next step.
    @return the predicted updated state.
    '''
    y = x.copy()
    y[0] += dt * x[3] * np.cos(y[2])
    y[1] += dt * x[3] * np.sin(y[2])
    y[2] += dt * x[4]
    return y


def HJacobian(x):
    '''
    Calculate the jacobian of the measurement model at x.

    @param x the 11 dimensional stat vector about which to approximate.
    '''
    return np.array(
        [[-np.cos(x[2]), -np.sin(x[2]), -np.sin(x[2]) * (x[5] - x[0]) + np.cos(x[2]) * (x[6] - x[1]), 0, 0, np.cos(x[2]), np.sin(x[2]), 0, 0, 0, 0],
         [np.sin(x[2]), -np.cos(x[2]), -np.cos(x[2]) * (x[5] - x[0]) - np.sin(x[2]) * (x[6] - x[1]), 0, 0, -np.sin(x[2]), np.cos(x[2]), 0, 0, 0, 0],
         [-np.cos(x[2]), -np.sin(x[2]), -np.sin(x[2]) * (x[7] - x[0]) + np.cos(x[2]) * (x[8] - x[1]), 0, 0, 0, 0, np.cos(x[2]), np.sin(x[2]), 0, 0],
         [np.sin(x[2]), -np.cos(x[2]), -np.cos(x[2]) * (x[7] - x[0]) - np.sin(x[2]) * (x[8] - x[1]), 0, 0, 0, 0, -np.sin(x[2]), np.cos(x[2]), 0, 0],
         [-np.cos(x[2]), -np.sin(x[2]), -np.sin(x[2]) * (x[9] - x[0]) + np.cos(x[2]) * (x[10] - x[1]), 0, 0, 0, 0, 0, 0, np.cos(x[2]), np.sin(x[2])],
         [np.sin(x[2]), -np.cos(x[2]), -np.cos(x[2]) * (x[9] - x[0]) - np.sin(x[2]) * (x[10] - x[1]), 0, 0, 0, 0, 0, 0, -np.sin(x[2]), np.cos(x[2])]
         ])


def hx(x):
    '''
    Convert from the state space to the measurement space.
    The measurement space is a 6 dimensional vector denoting the
    Cartesian locations of the landmarks relative to the robot.
    
    @param x the state to convert to a measurement.
    @return the corresponding measurement to the state.
    '''
    measurement = x[-6:].reshape(-1, 2)
    rotation = np.array([[np.cos(x[2]), np.sin(x[2])],
                         [-np.sin(x[2]), np.cos(x[2])]])
    translated = measurement - x[:2]
    rotated = np.dot(translated, rotation.T)
    return rotated.ravel()


def kalman(x, P, measurement, R, Q):
    '''
    Find the next estimated state and confidence.

    @param x the prior state.
    @param P the prior state covariance.
    @param R the measurement confidence. This is a 6x6 matrix.
    @param Q the prediction confidence. This is an 11x11 matrix.
    @return the updated state and confidence.
    '''
    F = FJacobian(x)

    xHat = fx(x)
    PHat = np.dot(np.dot(F, P), F.T) + Q
    H = HJacobian(xHat)

    y = measurement - hx(xHat)
    S = np.dot(np.dot(H, PHat), H.T) + R
    K = np.dot(np.dot(PHat, H.T), np.linalg.inv(S))

    xNew = xHat + np.dot(K, y)
    PNew = np.dot(np.eye(K.shape[0]) - np.dot(K, H), PHat)

    return xNew, PNew


# Calculate Ground Truth
ts = np.arange(0, 1.5 * np.pi, dt)
xs = 2 * (1 - np.cos(ts)) * np.cos(ts)
ys = 2 * (1 - np.cos(ts)) * np.sin(ts)
dxs = -np.sin(ts) + 2 * np.cos(ts) * np.sin(ts)
dys = np.cos(ts) + np.sin(ts) ** 2 - np.cos(ts) ** 2
thetas = np.arctan2(dxs, dys)

points = np.stack([xs, ys])

noise = 0.01
beaconNoise = 1
mean = 0
beacons = np.random.normal(mean, beaconNoise, (3, 2))

# Measurements of the landmarks translated to be relative to the robot.
# These are not yet rotated.
measureA = beacons[0, :, np.newaxis] - points
measureB = beacons[1, :, np.newaxis] - points
measureC = beacons[2, :, np.newaxis] - points

# Noisy measurements
noisyA = measureA + np.random.normal(mean, noise, measureA.shape)
noisyB = measureB + np.random.normal(mean, noise, measureB.shape)
noisyC = measureC + np.random.normal(mean, noise, measureC.shape)

measurements = np.vstack([noisyA, noisyB, noisyC]).T


def rotateMeasurement(theta, measurement):
    reshaped = measurement.reshape(-1, 2)
    rotation = np.array([[np.cos(theta), np.sin(theta)],
                         [-np.sin(theta), np.cos(theta)]])
    rotated = np.dot(reshaped, rotation.T)
    return rotated.ravel()


def kalmanSim():
    x = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    P = np.diag([0, 0, 1, 0.1, 0.05, beaconNoise, beaconNoise,
                 beaconNoise, beaconNoise, beaconNoise, beaconNoise])
    R = np.eye(6) * noise
    Q = np.diag([.1, .1, (np.pi / 18)**2, 0.1,
                 (np.pi/18)**2, .0, .0, .0, .0, .0, .0])

    result = []
    ps = []
    for i, measurement in enumerate(measurements):
        x, P = kalman(x, P, rotateMeasurement(thetas[i], measurement), R, Q)
        result.append(x)
        ps.append(P)

    return np.array(result).T, np.array(ps)


simRes, Ps = kalmanSim()

ax = plt.gca()
ax.set_aspect('equal')
ax.plot(xs, ys, c='r')
ax.plot(simRes[0], simRes[1], c='y')
ax.plot(simRes[5], simRes[6], c='g')
ax.plot(simRes[7], simRes[8], c='indigo')
ax.plot(simRes[9], simRes[10], c='orange')
ax.scatter(beacons[:, 0], beacons[:, 1], marker=(5, 1), s=200, c='r')


def errEllipse(mu, cov, factor=1):
    U, S, V = np.linalg.svd(cov)
    theta = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    return Ellipse(xy=mu,
                   width=2 * np.sqrt(S[0]) * factor,
                   height=2 * np.sqrt(S[1]) * factor,
                   angle=theta,
                   edgecolor='black')


ax.legend(['Ground Truth', 'Predicted Odometry', 'Predicted Beacon 1',
           'Predicted Beacon 2', 'Predicted Beacon 3', 'Actual Beacons'])

for i in range(0, Ps.shape[0], 5):
    ellipse = errEllipse(simRes[:2, i], Ps[i], factor=0.5)
    ax.add_patch(ellipse)

ax.set_xlabel('x position (m)')
ax.set_ylabel('y position (m)')
ax.figure.savefig('comparison.png')
plt.show()
