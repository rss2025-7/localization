from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped

from rclpy.node import Node
import rclpy
import numpy as np
from scipy import signal

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
        self.declare_parameter('num_particles', 100)

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.laser_sub = self.create_subscription(LaserScan, scan_topic,
                                                  self.laser_callback,
                                                  1)

        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 1)
        self.num_particles = self.get_parameter("num_particles").get_parameter_value().integer_value

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
        self.N = self.num_particles
        self.particles = np.zeros(self.N, 3)
        self.weights = np.ones(self.N)
        self.average = None

    def laser_callback(self, sensor_msg):
        resample_prob = np.random.binomial(n=1, p = 0.1)
        lidar_data = signal.decimate(sensor_msg.ranges,10) ###Downsample 1000 ---> 100
        new_weights = self.sensor_model.evaluate(self.particles, lidar_data)
        if resample_prob == 1:
            particle_indices = np.random.choice(self.N, self.N, p = new_weights)
            self.weights = np.ones(self.N)
            self.particles = self.particles[particle_indices, :]
            best_idx = np.argmax(new_weights)
            best_particle = self.particles[best_idx, :]
        else:
            self.weights *= new_weights
            best_idx = np.argmax(self.weights)
            best_particle = self.particles[best_idx, :]

        self.average = best_particle
        


    def odom_callback(self, odom_msg):
        delta = odom_msg.twist
        odom_velo = [delta[0], delta[7], delta[-1]]
        self.particles = self.motion_model.evaluate(self.particles, odom_velo)
        self.average = self.particles[0,:]



def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
