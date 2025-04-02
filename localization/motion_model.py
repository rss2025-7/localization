import numpy as np

class MotionModel:

    def __init__(self, node):
        ####################################
        self.sigma_dx = 0.05         
        self.sigma_dy = 0.05        
        self.sigma_dtheta = np.deg2rad(1)   
        self.deterministic = False
        ####################################

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:
                [x0, y0, theta0]
                [x1, y1, theta1]
                [   ...      ]
            odometry: A 3-vector [dx, dy, dtheta]
                      where dx, dy are in the robot's local frame

        returns:
            particles: An updated matrix of the same size
        """
        ####################################
        dx, dy, dtheta = odometry

        N = particles.shape[0]
        if not self.deterministic:
            noisy_dx = dx + np.random.normal(0, self.sigma_dx, size=N)
            noisy_dy = dy + np.random.normal(0, self.sigma_dy, size=N)
            noisy_dtheta = dtheta + np.random.normal(0, self.sigma_dtheta, size=N)
        else:
            noisy_dx, noisy_dy, noisy_dtheta = dx,dy, dtheta
        ####################################
        
        theta = particles[:, 2]
        particles[:, 0] += np.cos(theta) * noisy_dx - np.sin(theta) * noisy_dy
        particles[:, 1] += np.sin(theta) * noisy_dx + np.cos(theta) * noisy_dy
        particles[:, 2] += noisy_dtheta
        
        return particles
