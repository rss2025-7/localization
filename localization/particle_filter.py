from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

import numpy as np

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseWithCovariance, PoseArray, Pose, TransformStamped
from sensor_msgs.msg import LaserScan
from tf_transformations import quaternion_from_euler

from tf2_ros import TransformBroadcaster

from rclpy.node import Node
import rclpy
import numpy as np
import scipy

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
        self.declare_parameter('num_particles', 200) # originally 100

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

        self.visual_pub = self.create_publisher(PoseArray, "/particle_viz", 1)

        self.tf_broadcaster = TransformBroadcaster(self)


        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        self.num_beams_per_particle = self.sensor_model.num_beams_per_particle

        # Deterministic motion model
        self.declare_parameter('deterministic', False)
        self.motion_model.deterministic = self.get_parameter('deterministic').get_parameter_value().bool_value
        self.sim = True

        # Particle filter parameters
        # self.declare_parameter('num_particles', 200)
        self.N = self.num_particles
        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value
        self.particles = np.zeros((self.num_particles, 3))
        self.weights = np.ones(self.N)
        self.last_time = self.get_clock().now()

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
        self.N = self.num_particles
        # self.particles = np.zeros(self.N, 3)
        # self.weights = np.ones(self.N)
        # self.average = None

    def laser_callback(self, sensor_msg):
        resample_prob = np.random.binomial(n=1, p = 0.5)

        downsampled_indices = np.linspace(0, len(sensor_msg.ranges) - 1, self.num_beams_per_particle).astype(int)
        lidar_data = np.array(sensor_msg.ranges)[downsampled_indices]

        new_weights = self.sensor_model.evaluate(self.particles, lidar_data)
        if new_weights is not None:
            if resample_prob == 0:
                new_weights = np.power(new_weights, 1/3)
                new_weights_sum = new_weights.sum()
                # self.get_logger().info(f"Weight sum: {new_weights_sum}")
                particle_indices = np.random.choice(self.N, self.N, p = new_weights/new_weights_sum)
                self.weights = new_weights[particle_indices]
                self.particles = self.particles[particle_indices, :]
                # best_idx = np.argmax(new_weights)
                # best_particle = self.particles[best_idx, :]
            else:
                # self.get_logger().info(f'Types: {type(self.weights), type(new_weights)}')
                self.weights *= new_weights
                # best_idx = np.argmax(self.weights)
                # best_particle = self.particles[best_idx, :]

            # self.odom_publisher()



    def odom_callback(self, odom_msg):
        odom_velo = odom_msg.twist.twist.linear
        xdot, ydot, thetadot = odom_velo.x, odom_velo.y, odom_msg.twist.twist.angular.z

        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9  # seconds
        self.last_time = current_time
        if self.sim:
            odom_factor = 1
        else:
            odom_factor = -1
        odom_info = odom_factor * np.array([xdot*dt, ydot*dt, thetadot*dt]) #Negative 1 for real robot

        self.particles = self.motion_model.evaluate(self.particles, odom_info)
        self.odom_publisher()


    def odom_publisher(self):
        odom = Odometry()
        odom.child_frame_id = self.particle_filter_frame # change for sim/real
        odom.header.frame_id = "map"
        odom.header.stamp = self.get_clock().now().to_msg()

        bestx = np.sum(self.weights * self.particles[:, 0]) / np.sum(self.weights)
        besty = np.sum(self.weights * self.particles[:, 1]) / np.sum(self.weights)

        # bestx, besty= np.average(self.particles[:,0]), np.average(self.particles[:,1]) # doesn't work

        best_theta = np.arctan2(np.sum(self.weights*np.sin(self.particles[:,2])), np.sum(self.weights*np.cos(self.particles[:,2])))
        best_particle = np.array([bestx, besty, best_theta])

        orientation = odom.pose.pose.orientation
        orientation.x, orientation.y, orientation.z, orientation.w = quaternion_from_euler(0, 0, best_particle[2])
        odom.pose.pose.orientation = orientation

        odom_pose = odom.pose.pose.position
        odom_pose.x, odom_pose.y, odom_pose.z = best_particle[0], best_particle[1], 0.0
        odom.pose.pose.position = odom_pose

        if not self.sim:
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "map"
            t.child_frame_id = self.particle_filter_frame

            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z  = bestx, besty, 0.0
            t.transform.rotation.x = orientation.x
            t.transform.rotation.y = orientation.y
            t.transform.rotation.z = orientation.z
            t.transform.rotation.w = orientation.w

            self.tf_broadcaster.sendTransform(t)

        # odom.twist = twist # don't know if necessary
        # self.get_logger().info(f"help")

        self.odom_pub.publish(odom)
        self.visualize()


    def quaternion_to_yaw(self, quaternion):
        """
        Convert a quaternion to yaw angle
        """
        qx = quaternion.x
        qy = quaternion.y
        qz = quaternion.z
        qw = quaternion.w
        return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))

    def particle2pose(self, particle):
        x,y,t = particle
        pose_msg = Pose()

        pose_msg.position.x = float(x)
        pose_msg.position.y = float(y)
        pose_msg.position.z = 0.0

        xq,yq,zq,wq = quaternion_from_euler(0.0,0.0,t)

        pose_msg.orientation.x = xq
        pose_msg.orientation.y = yq
        pose_msg.orientation.z = zq
        pose_msg.orientation.w = wq

        return pose_msg


    def visualize(self):
        msg = PoseArray()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.poses = [self.particle2pose(part) for part in self.particles]

        self.visual_pub.publish(msg)
        # self.get_logger().info(f"{self.particles}")


def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
