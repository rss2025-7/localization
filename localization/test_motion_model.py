from localization.motion_model import MotionModel
from rclpy.node import Node
import rclpy
import numpy as np
import matplotlib.pyplot as plt

class MotionModelTest(Node):

    def __init__(self):
        super().__init__("motion_testor")
        self.model = MotionModel(self)
        print("--------BEGINNING MOTION MODEL TEST--------")
        print("--------NO NOISE--------")
        self.test_no_noise()
        print("--------WITH NOISE--------")
        self.test_noise()
        print("--------WITH MUCH NOISE--------")
        self.test_big_noise()

    def test_no_noise(self):
        particles = np.zeros((10, 3))
        points = []
        points.append(np.copy(particles))
        odom_list = [[1, 0, 0],
                     [1, 0, 0],
                     [0, 1, np.deg2rad(90)],
                     [1, 0, 0],]
        self.model.deterministic = True
        for odom in odom_list:
            print("")
            particles = self.model.evaluate(particles, odom)
            points.append(np.copy(particles))
        print(repr(np.array(points)))

    def test_noise(self):
        particles = np.zeros((10, 3))
        points = []
        points.append(np.copy(particles))
        odom_list = [[1, 0, 0],
                     [1, 0, 0],
                     [0, 1, np.deg2rad(90)],
                     [1, 0, 0],]
        self.model.deterministic = False
        self.model.sigma_dx = 0.1
        self.model.sigma_dy = 0.1
        self.model.sigma_dtheta = np.deg2rad(5)
        for odom in odom_list:
            print("")
            particles = self.model.evaluate(particles, odom)
            points.append(np.copy(particles))
        print(repr(np.array(points)))
    
    def test_big_noise(self):
        particles = np.zeros((10, 3))
        points = []
        points.append(np.copy(particles))
        odom_list = [[1, 0, 0],
                     [1, 0, 0],
                     [0, 1, np.deg2rad(90)],
                     [1, 0, 0],]
        self.model.deterministic = False
        self.model.sigma_dx = 0.5
        self.model.sigma_dy = 0.5
        self.model.sigma_dtheta = np.deg2rad(20)
        for odom in odom_list:
            print("")
            particles = self.model.evaluate(particles, odom)
            points.append(np.copy(particles))
        print(repr(np.array(points)))

def main(args=None):
    rclpy.init(args=args)
    tester = MotionModelTest()
    rclpy.spin(tester)
    rclpy.shutdown()
