# /usr/bin/env python
import rospy
import numpy as np
from geometry_msgs.msg import PoseWithCovariance
from tf.transformations import quaternion_from_euler
from itertools import permutations

from sensor_msgs.point_cloud2 import read_points, create_cloud_xyz32
from sensor_msgs.msg import PointCloud2


class RosSlam:
    def __init__(self,
                 R=np.eye(6) * 0.005,
                 Q=np.diag([.1, .1, (np.pi / 18)**2, 0.1,
                            (np.pi/18)**2, .0, .0, .0, .0, .0, .0]),
                 P=np.diag([0, 0, 1, 0.1, 0.05, 1, 1, 1, 1, 1, 1])):

        self.R = R
        self.Q = Q
        self.P = P
        self.x = np.zeros(11)
        self.t0 = rospy.Time.now()

        self.posePub = rospy.publisher(
            '/pose', PoseWithCovariance, queue_size=1)

        self.landmarkPub = rospy.publisher(
            '/predicted_landmarks', PointCloud2, queue_size=1)
        self.scanSub = rospy.Subscriber(
            '/measured_landmarks', PointCloud2, self.kalmanCb)

    def HJacobian(self, x):
        '''
        Calculate the jacobian of the measurement model at x.

        @param x the 11 dimensional stat vector about which to approximate.
        '''
        return np.array(
            [[-np.cos(x[2]), -np.sin(x[2]), -np.sin(x[2]) * (x[5] - x[0]) + np.cos(x[2]) * (x[6] - x[1]), 0, 0, np.cos(x[2]), np.sin(x[2]), 0, 0, 0, 0],
             [np.sin(x[2]), -np.cos(x[2]), -np.cos(x[2]) * (x[5] - x[0]) - np.sin(x[2])
              * (x[6] - x[1]), 0, 0, -np.sin(x[2]), np.cos(x[2]), 0, 0, 0, 0],
             [-np.cos(x[2]), -np.sin(x[2]), -np.sin(x[2]) * (x[7] - x[0]) + np.cos(x[2])
              * (x[8] - x[1]), 0, 0, 0, 0, np.cos(x[2]), np.sin(x[2]), 0, 0],
             [np.sin(x[2]), -np.cos(x[2]), -np.cos(x[2]) * (x[7] - x[0]) - np.sin(x[2])
              * (x[8] - x[1]), 0, 0, 0, 0, -np.sin(x[2]), np.cos(x[2]), 0, 0],
             [-np.cos(x[2]), -np.sin(x[2]), -np.sin(x[2]) * (x[9] - x[0]) + np.cos(x[2])
              * (x[10] - x[1]), 0, 0, 0, 0, 0, 0, np.cos(x[2]), np.sin(x[2])],
             [np.sin(x[2]), -np.cos(x[2]), -np.cos(x[2]) * (x[9] - x[0]) - np.sin(x[2])
              * (x[10] - x[1]), 0, 0, 0, 0, 0, 0, -np.sin(x[2]), np.cos(x[2])]
             ])

    def FJacobian(self, x, dt):
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

    def hx(self, x):
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

    def fx(self, x, dt):
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

    def kalmanCb(self, data):
        measurement = np.array([(p.x, p.y) for p in data.points]).ravel()

        dt = (data.header.stamp - self.t0).to_sec()
        self.t0 = data.header.stamp

        # Predict
        F = self.FJacobian(self.x, dt)

        xHat = self.fx(self.x, dt)
        PHat = np.dot(np.dot(F, self.P), F.T) + self.Q

        # Update
        y = measurement - self.hx(xHat)

        H = self.HJacobian(xHat)
        S = np.dot(np.dot(H, PHat), H.T) + self.R
        K = np.dot(np.dot(PHat, H.T), np.linalg.inv(S))

        self.x = xHat + np.dot(K, y)
        self.P = np.dot(np.eye(K.shape[0]) - np.dot(K, H), PHat)

        # Publish
        pose = PoseWithCovariance()
        pose.pose.position.x = self.x[0]
        pose.pose.position.y = self.x[1]
        q = quaternion_from_euler(0, 0, self.x[2])
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        covariance = np.insert(self.P[:3, :3], [2, 2, 2], 0, axis=0)
        covariance = np.insert(covariance, [2, 2, 2], 0, axis=1)
        pose.covariance = covariance.ravel()

        self.posePub.publish(pose)

        cloud = create_cloud_xyz32(
            rospy.Header(frame_id='odom', stamp=rospy.Time.now()),
            self.x[-6:].reshape(3, 2))
        self.landmarkPub.publish(cloud)
