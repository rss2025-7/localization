import numpy as np
from scan_simulator_2d import PyScanSimulator2D
# Try to change to just `from scan_simulator_2d import PyScanSimulator2D`
# if any error re: scan_simulator_2d occurs

from tf_transformations import euler_from_quaternion

from nav_msgs.msg import OccupancyGrid

import sys

np.set_printoptions(threshold=sys.maxsize)


class SensorModel:

    def __init__(self, node):
        node.declare_parameter('map_topic', "default")
        node.declare_parameter('num_beams_per_particle', 1)
        node.declare_parameter('scan_theta_discretization', 1.0)
        node.declare_parameter('scan_field_of_view', 1.0)
        node.declare_parameter('lidar_scale_to_map_scale', 1.0)

        self.map_topic = node.get_parameter('map_topic').get_parameter_value().string_value
        self.num_beams_per_particle = node.get_parameter('num_beams_per_particle').get_parameter_value().integer_value
        self.scan_theta_discretization = node.get_parameter(
            'scan_theta_discretization').get_parameter_value().double_value
        self.scan_field_of_view = node.get_parameter('scan_field_of_view').get_parameter_value().double_value
        self.lidar_scale_to_map_scale = node.get_parameter(
            'lidar_scale_to_map_scale').get_parameter_value().double_value

        ####################################
        # Adjust these parameters
        self.alpha_hit = 0.74
        self.alpha_short = 0.07
        self.alpha_max = 0.07
        self.alpha_rand = 0.12
        self.sigma_hit = 8.0

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201
        self.z_max = 200.0
        self.z_min = 0.0
        ####################################

        node.get_logger().info("%s" % self.map_topic)
        node.get_logger().info("%s" % self.num_beams_per_particle)
        node.get_logger().info("%s" % self.scan_theta_discretization)
        node.get_logger().info("%s" % self.scan_field_of_view)
        self.node = node

        # Precompute the sensor model table
        self.sensor_model_table = np.empty((self.table_width, self.table_width))
        self.precompute_sensor_model()
        # self.node.get_logger().info(f'PRECOMPUTE: {self.sensor_model_table}')


        # Create a simulated laser scan
        self.scan_sim = PyScanSimulator2D(
            self.num_beams_per_particle,
            self.scan_field_of_view,
            0,  # This is not the simulator, don't add noise
            0.01,  # This is used as an epsilon
            self.scan_theta_discretization)

        # Subscribe to the map
        self.map = None
        self.map_set = False
        self.map_subscriber = node.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            1)

    def precompute_sensor_model(self):
        """
        Generate and store a table which represents the sensor model.

        For each discrete computed range value, this provides the probability of
        measuring any (discrete) range. This table is indexed by the sensor model
        at runtime by discretizing the measurements and computed ranges from
        RangeLibc.
        This table must be implemented as a numpy 2D array.

        Compute the table based on class parameters alpha_hit, alpha_short,
        alpha_max, alpha_rand, sigma_hit, and table_width.

        args:
            N/A

        returns:
            No return type. Directly modify `self.sensor_model_table`.
        """
        d = np.linspace(self.z_min, self.z_max, self.table_width)
        row = np.linspace(self.z_min, self.z_max, self.table_width)
        z_ks = np.tile(row[:, np.newaxis], (1, self.table_width))
        self.z_max = np.max(z_ks)

        p_hit_coeff = 1 / np.sqrt(2 * np.pi * self.sigma_hit**2)
        p_hit = np.where((z_ks >= 0) & (z_ks <= self.z_max), p_hit_coeff * np.exp(-(z_ks-d)**2 / (2*self.sigma_hit**2)), 0.0)
        p_hit_row_sums = p_hit.sum(axis=0, keepdims=True)
        p_hit = np.where(p_hit_row_sums != 0, p_hit / p_hit_row_sums, 0)

        p_short = np.where((z_ks >= 0) & (z_ks <= d) & (d != 0), (2/np.where(d != 0, d, 1)) * (1 - z_ks/np.where(d != 0, d, 1)), 0.0)
        p_max = np.where((z_ks == self.z_max), 1.0, 0.0)
        p_rand = np.where((z_ks >= 0) & (z_ks <= self.z_max), 1/self.z_max, 0.0)

        self.sensor_model_table = self.alpha_hit * p_hit + \
                                  self.alpha_short * p_short + \
                                  self.alpha_max * p_max + \
                                  self.alpha_rand * p_rand

        # self.node.get_logger().info(f'OG BEFORE NORM: {self.sensor_model_table}')
        normalized_table_sums = self.sensor_model_table.sum(axis=0, keepdims=True)
        self.sensor_model_table = np.where(normalized_table_sums != 0, self.sensor_model_table / normalized_table_sums, 0)

    def evaluate(self, particles, observation):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            observation: A vector of lidar data measured
                from the actual lidar. THIS IS Z_K. Each range in Z_K is Z_K^i

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """

        # self.node.get_logger().info(f'ENTERED SENSOR MODEL EVALUATE')

        if not self.map_set:
            # self.node.get_logger().info(f'EXITED')
            return

        ####################################
        # TODO
        # Evaluate the sensor model here!
        #
        # You will probably want to use this function
        # to perform ray tracing from all the particles.
        # This produces a matrix of size N x num_beams_per_particle

        scans = self.scan_sim.scan(particles)
        observation /= self.resolution * self.lidar_scale_to_map_scale
        scans /= self.resolution * self.lidar_scale_to_map_scale

        observation = observation.astype(int)
        scans = scans.astype(int)

        observation = np.clip(observation, a_min=int(0), a_max=int(self.z_max))
        scans = np.clip(scans, a_min=int(0), a_max=int(self.z_max))

        probabilities = []
        for particle_ground_truths in scans:
            indices = np.array(list(zip(observation, particle_ground_truths)))
            probability = self.sensor_model_table[indices[:,0], indices[:,1]]
            # self.node.get_logger().info(f'SENSOR ORIGINAL PROBABILITIES: {len(probabilities)}, {np.array(probabilities)}')
            probability = np.prod(probability)
            probabilities.append(probability)

        # self.node.get_logger().info(f'SENSOR MODEL: {len(probabilities)}, {np.array(probabilities)}')
        return np.array(probabilities)

        # observation_broadcasted = np.tile(observation, (scans.shape[0], 1))
        # probabilities_array = self.sensor_model_table[observation_broadcasted, scans]
        # probabilities = np.prod(probabilities_array, axis=1)

        return probabilities
        ####################################

    def map_callback(self, map_msg):
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double) / 100.
        self.map = np.clip(self.map, 0, 1)

        self.resolution = map_msg.info.resolution

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation
        origin_o = euler_from_quaternion((
            origin_o.x,
            origin_o.y,
            origin_o.z,
            origin_o.w))
        origin = (origin_p.x, origin_p.y, origin_o[2])

        # Initialize a map with the laser scan
        self.scan_sim.set_map(
            self.map,
            map_msg.info.height,
            map_msg.info.width,
            map_msg.info.resolution,
            origin,
            0.5)  # Consider anything < 0.5 to be free

        # Make the map set
        self.map_set = True

        print("Map initialized")
