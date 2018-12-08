# /usr/bin/env python
import rospy
import numpy as np
from geometry_msgs.msg import PoseWithCovariance
from tf.transformations import quaternion_from_euler
from itertools import permutations

from sensor_msgs.point_cloud2 import read_points, create_cloud_xyz32
from sensor_msgs.msg import PointCloud2


def reorderPoints(reference, points):
    '''
    Sort the points to least-squares resemble the reference points

    @param x the pose of the object
    @param points array of shape (n, 2) denoting the locations
    of the beacons

    @return points ordered by least squared distance from
    corresponding stored points

    Example:
    >>> reference = np.array([[1,2], [10,9], [5,4]])
    >>> points = np.array([[4, 5], [3, 2], [8,7]])
    >>> reorderPoints(reference, points)
    Out:
    array([[3, 2],
           [8, 7],
           [4, 5]])
    '''

    perms = np.array(list(permutations(range(points.shape[0]))))
    relativeCoords = reference - points[perms]
    ssd = (relativeCoords**2).sum((2, 1))

    return points[perms[ssd.argmin()]]


class RosSlam:
    def __init__(self,
                 R=np.eye(6) * 0.005,
                 Q=np.diag(0.005, 0.005, 0.001, 0.005,
                           0.001, 0, 0, 0, 0, 0, 0),
                 P=np.diag([0, 0, 1, .001, 0.001, 2, 2, 2, 2, 2, 2])):

        self.R = R
        self.Q = Q
        self.P = P
        self.x = np.zeros(11)
        self.t0 = rospy.Time.now()
        self.H = np.array(
            [[-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
             [0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
             [-1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
             [0, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
             [-1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
             [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

        self.posePub = rospy.publisher(
            '/pose', PoseWithCovariance, queue_size=1)

        self.landmarkPub = rospy.publisher(
            '/predicted_landmarks', PointCloud2, queue_size=1)
        self.scanSub = rospy.Subscriber(
            '/measured_landmarks', PointCloud2, self.kalmanCb)

    def reorderLandmarks(x, landmarks):
        '''
        Sort the landmarks to least-squares resemble the existing landmarks

        @param x the pose of the object
        @param landmarks array of shape (n, 2) denoting the locations
        of the beacons

        @return landmarks ordered by least squared distance from
        corresponding stored landmarks
        '''

        rotation = np.array(
            [[np.cos(x[2]), -np.sin(x[2])],
             [np.sin(x[2]), np.cos(x[2])]])

        rotated = np.dot(landmarks, rotation.T)
        translated = rotated + x[:2]

        return reorderPoints(x[-6:].reshape(-1, 2), translated).ravel()

    def FJacobian(self, dt):
        return np.array(
            [[1, 0, - self.x[3] * np.sin(self.x[2]) * dt,
              np.cos(self.x[2]) * dt, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, self.x[3] * np.cos(self.x[2]) * dt, np.sin(self.x[2])
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

    def kalmanCb(self, data):
        measurement = self.reorderLandmarks(
            self.x, np.array(list(read_points(data)))[:, :2])

        dt = (data.header.stamp - self.t0).to_sec()
        self.t0 = data.header.stamp

        # Predict
        F = self.FJacobian(self.x, dt)

        xHat = np.dot(F, self.x)
        PHat = np.dot(np.dot(F, self.P), F.T) + self.Q

        # Update
        y = measurement - np.dot(self.H, xHat)
        S = np.dot(np.dot(self.H, PHat), self.H.T) + self.R
        K = np.dot(np.dot(PHat, self.H.T), np.linalg.inv(S))

        self.x = xHat + np.dot(K, y)
        self.P = np.dot(np.eye(K.shape[0]) - np.dot(K, self.H), PHat)

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
