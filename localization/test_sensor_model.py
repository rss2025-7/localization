from localization.sensor_model import SensorModel
from rclpy.node import Node
import rclpy
import numpy as np
import matplotlib.pyplot as plt

class ScanSim:
    def __init__(self):
        pass
    def scan(self, particles):
        return particles

class SensorModelTest(Node):

    def __init__(self):
        super().__init__("sensor_tester")
        self.model = SensorModel(self)
        self.model.map_set = True
        self.model.scan_sim = ScanSim()
        self.model.resolution = 1
        self.model.lidar_scale_to_map_scale = 1
        print("--------BEGINNING SENSOR MODEL TEST--------")
        self.test_sensor_model()

    def test_sensor_model(self):
        results = []
        for i in range(10):
            observation = np.random.randint(0, 200, size=200).astype(float)
            noise_scale_01 = self.create_similar_array(observation, 0.1)
            noise_scale_05 = self.create_similar_array(observation, 0.5)
            noise_scale_10 = self.create_similar_array(observation, 1)
            noise_scale_15 = self.create_similar_array(observation, 1.5)
            noise_scale_20 = self.create_similar_array(observation, 2)
            noise_scale_100 = self.create_similar_array(observation, 10)

            particles = np.array([
                noise_scale_01,            
                noise_scale_05,
                noise_scale_10,
                noise_scale_15,
                noise_scale_20,
                noise_scale_100,
            ])
            result = self.model.evaluate(particles, observation)
            result_sum = np.sum(result)
            result = result / result_sum
            results.append(result)
        results = np.array(results)
        print(str(results))
        # results = np.mean(results, axis=0)
        plt.boxplot(results, positions=[0.1, 0.5, 1, 1.5, 2, 10])

        # Label x-axis
        plt.xlabel("x value index")
        plt.ylabel("Value")
        plt.title("Boxplot of each x value (per column)")
        plt.xticks([0.1, 0.5, 1, 1.5, 2, 10])  # Boxplot indices start at 1

        plt.show()

    def create_similar_array(self, arr, percent_same):
        N = len(arr)
        similar_arr = np.copy(arr)
        similar_arr += np.random.normal(0, percent_same, size=N)
        # different_indices = np.random.choice(N, size=int((1-percent_same) * N), replace=False)
        # similar_arr[different_indices] = np.random.randint(0, 200, size=len(different_indices))
        return similar_arr
    
def main(args=None):
    rclpy.init(args=args)
    tester = SensorModelTest()
    rclpy.spin(tester)
    rclpy.shutdown()
