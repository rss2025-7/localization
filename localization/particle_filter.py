from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

import numpy as np

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped

from rclpy.node import Node
import rclpy

assert rclpy


class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('particle_filter_frame', "default")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value

        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.

        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.laser_sub = self.create_subscription(LaserScan, scan_topic,
                                                  self.laser_callback,
                                                  1)

        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        # Particle filter parameters
        self.num_particles = 200 # Adjustable
        self.particles = np.zeros((self.num_particles, 3))

        self.get_logger().info("=============+READY+=============")

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.
    def pose_callback(self, msg):
        """
        initialize particles
        """
        pose = msg.pose.pose # pose type
        x, y = pose.position.x, pose.position.y
        theta = self.quaternion_to_yaw(pose.orientation)

        # x, y, z, rotation about x, y, z
        covariance_matrix = msg.pose.covariance

        sigma_x = np.sqrt(covariance_matrix[0])
        sigma_y = np.sqrt(covariance_matrix[7])
        sigma_theta = np.sqrt(covariance_matrix[35])
        # Initialize particles around intiail pose
        self.particles[:, 0] = np.random.normal(x, sigma_x, self.num_particles)
        self.particles[:, 1] = np.random.normal(y, sigma_y, self.num_particles)
        self.particles[:, 2] = np.random.normal(theta, sigma_theta, self.num_particles)

    def quaternion_to_yaw(self, quaternion):
        """
        Convert a quaternion to yaw angle
        """
        qx = quaternion.x
        qy = quaternion.y
        qz = quaternion.z
        qw = quaternion.w
        return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))

def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
