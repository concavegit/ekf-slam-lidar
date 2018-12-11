import rospy
import numpy as np
import random
from sensor_msgs.msgs import LaserScan, PointCloud
from geometry_msgs.msg import Point


def pol2cart(r, theta):
    return r * np.cos(theta), r * np.sin(theta)


def circle3(data):
    '''Find the center of a circle from 3 points'''
    deltas = (data[:, 1:] - data[:, :-1])
    slopes = deltas[1] / deltas[0]
    cx = (slopes.prod() * (data[1, 0] - data[1, 2]) + slopes[1] *
          (data[0, 0] + data[0, 1]) - slopes[0] * (data[0, 1] + data[0, 2]))\
        / 2 / (slopes[1] - slopes[0])
    cy = ((data[0, 0] + data[0, 1]) / 2 - cx) / \
        slopes[0] + (data[1, 0] + data[1, 1]) / 2
    center = np.array([cx, cy])
    radius = np.sqrt(((data[:, 0] - center) ** 2).sum())
    return center, radius


def rcirc(data, r, k, d, minPts):
    '''RANSAC data for a circle of radius r.'''
    if k < 0 or data.shape[1] < minPts:
        return np.array([]), data

    pt1 = data[:, np.random.randint(data.shape[1])]

    candidates = data[:, np.linalg.norm(
        data - pt1[:, np.newaxis], axis=0) < 2 * (r + d)]

    if candidates.shape[1] < minPts:
        return rcirc(data, r, k - 1, d, minPts)

    c, r1 = circle3(candidates[:, sorted(
        random.sample(range(candidates.shape[1]), 3))])

    if np.abs(r1 - r) > .05:
        return rcirc(data, r, k - 1, d, minPts)

    mask = np.abs(np.linalg.norm((data - c[:, np.newaxis]), axis=0) - r1) < d

    if mask.sum() < minPts:
        return rcirc(data, r, k - 1, d, minPts)

    return c, data[:, ~mask]


class getBuckets:
    def __init__(self):
        rospy.init_node('getBucket')
        self.subScan = rospy.Subscriber('/scan', LaserScan, self.scanCallback)
        self.pubLandmarks = rospy.Publisher(
            'landmarks', PointCloud, queue_size=10)

    def scanCallback(self, data):
        ranges = np.array(data.ranges)
        thetas = np.arange(0, 360) * np.pi / 180
        points = np.array(pol2cart(ranges, thetas))
        circ1 = rcirc(points, 0.11, 512, 0.005, 6)[0]
        circ2 = rcirc(points, 0.08, 512, 0.005, 6)[0]
        circ3 = rcirc(points, 0.06, 512, 0.005, 6)[0]

        p1 = Point()
        p1.x, p1.y = circ1
        p2 = Point()
        p2.x, p2.y = circ2
        p3 = Point()
        p3.x, p3.y = circ3

        pcl = PointCloud()
        pcl.points = [p1, p2, p3]
        pcl.header.time = rospy.Time.now()

        self.pubLandmarks.publish(pcl)
