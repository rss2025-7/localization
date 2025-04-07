import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np

class AckermannDrivePublisher(Node):
    def __init__(self):
        super().__init__('test_drive')

        # Declare parameters for base velocity and angular velocity
        self.declare_parameter('base_velocity', 1.0)  # Base linear velocity
        self.declare_parameter('base_angular_velocity', 0.0)  # Base angular velocity
        self.declare_parameter('noise_std_dev', 0.1)  # Standard deviation for noise

        # Retrieve parameters
        self.base_velocity = self.get_parameter('base_velocity').get_parameter_value().double_value
        self.base_angular_velocity = self.get_parameter('base_angular_velocity').get_parameter_value().double_value
        self.noise_std_dev = self.get_parameter('noise_std_dev').get_parameter_value().double_value

        # Create a publisher for AckermannDrive messages
        self.publisher = self.create_publisher(AckermannDriveStamped, 'drive', 10)

        # Create a timer to publish messages at a fixed rate
        self.timer = self.create_timer(0.1, self.publish_ackermann_drive)  # Publish at 10 Hz

    def publish_ackermann_drive(self):
        # Generate random noise
        # velocity_noise = np.random.normal(0, self.noise_std_dev)
        # angular_velocity_noise = np.random.normal(0, self.noise_std_dev)

        velocity_noise = np.random.normal(0, self.noise_std_dev)
        angular_velocity_noise = np.random.normal(0, self.noise_std_dev)

        # Add noise to base velocity and angular velocity
        noisy_velocity = self.base_velocity + velocity_noise
        noisy_angular_velocity = self.base_angular_velocity + angular_velocity_noise

        # Create and populate the AckermannDriveStamped message
        msg = AckermannDriveStamped()
        msg.drive.speed = noisy_velocity
        msg.drive.steering_angle = noisy_angular_velocity
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.drive.acceleration = 0.0
        msg.drive.jerk = 0.0
        msg.drive.steering_angle_velocity = 0.0
        # Publish the message
        self.publisher.publish(msg)
        self.get_logger().info(f'Published: speed={noisy_velocity:.2f}, steering_angle={noisy_angular_velocity:.2f}')

def main(args=None):
    rclpy.init(args=args)
    node = AckermannDrivePublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
