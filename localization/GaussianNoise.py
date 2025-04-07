import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import random

class GaussianNoise(Node):

    def __init__(self):
        super().__init__("gaussian_noise")

        self.declare_parameter('linear_noise_std', 0.9)
        self.declare_parameter('angular_noise_std', 0.85)

        self.linear_noise_std = self.get_parameter('linear_noise_std').get_parameter_value().double_value
        self.angular_noise_std = self.get_parameter('angular_noise_std').get_parameter_value().double_value

        self.declare_parameter('odom_topic', "/odom")
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value


        self.odom_subscriber = self.create_subscription(Odometry, odom_topic, self.odom_callback, 1)
        self.odom_noisy_publisher = self.create_publisher(Odometry, 'odom_noisy', 1)

    def odom_callback(self, odom_msg):
        vx = odom_msg.twist.covariance[0]
        vy = odom_msg.twist.covariance[7]
        vth = odom_msg.twist.covariance[-1]

        vx_noisy = vx + random.gauss(0, self.linear_noise_std)
        vy_noisy = vy + random.gauss(0, self.linear_noise_std)
        vth_noisy = vth + random.gauss(0, self.angular_noise_std)

        noisy_msg = Odometry()
        noisy_msg.header = odom_msg.header
        noisy_msg.child_frame_id = odom_msg.child_frame_id

        # Keep pose if needed, or set it to msg.pose
        noisy_msg.pose = odom_msg.pose

        noisy_msg.twist.covariance[0] = vx_noisy
        noisy_msg.twist.covariance[7] = vy_noisy
        noisy_msg.twist.covariance[-1] = vth_noisy

        self.odom_noisy_publisher.publish(noisy_msg)

def main(args=None):
    rclpy.init(args=args)
    node = GaussianNoise()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
