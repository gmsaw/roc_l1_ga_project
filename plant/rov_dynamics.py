"""
ROV 6-DOF Dynamics Model
Implements mass, Coriolis, and damping matrices for underwater vehicle
"""

import numpy as np
from scipy.integrate import odeint


class ROVDynamics:
    """6-DOF ROV Dynamics Model"""
    
    def __init__(self):
        """Initialize ROV parameters and state"""
        # Mass and inertia (simplified for 6 DOF)
        # m: mass (kg), I: moments of inertia (kg*m^2)
        self.m = 100.0
        self.Ix = 10.0
        self.Iy = 10.0
        self.Iz = 15.0
        
        # Mass matrix (including added mass)
        # Simplified: diagonal matrix for uncoupled dynamics
        self.M = np.diag([
            self.m + 20.0,      # X direction added mass
            self.m + 20.0,      # Y direction added mass  
            self.m + 30.0,      # Z direction added mass
            self.Ix + 5.0,      # Roll added inertia
            self.Iy + 5.0,      # Pitch added inertia
            self.Iz + 8.0       # Yaw added inertia
        ])
        
        # Linear damping matrix (simplified)
        self.D = np.diag([
            50.0,   # X damping
            50.0,   # Y damping
            80.0,   # Z damping
            20.0,   # Roll damping
            20.0,   # Pitch damping
            25.0    # Yaw damping
        ])
        
        # Nonlinear damping coefficients (quadratic)
        self.D_nl = np.diag([
            30.0,   # X quadratic damping
            30.0,   # Y quadratic damping
            40.0,   # Z quadratic damping
            10.0,   # Roll quadratic damping
            10.0,   # Pitch quadratic damping
            12.0    # Yaw quadratic damping
        ])
        
        # Coriolis and centripetal matrix (simplified)
        # Will be computed based on velocity
        
        # State: [x, y, z, roll, pitch, yaw, u, v, w, p, q, r]
        self.state = np.zeros(12)
        
        # Control inputs: [Fx, Fy, Fz, tau_roll, tau_pitch, tau_yaw]
        self.control_input = np.zeros(6)
        
    def reset(self):
        """Reset ROV state to initial conditions"""
        self.state = np.zeros(12)
        self.control_input = np.zeros(6)
        
    def get_state(self):
        """Return current state vector"""
        return self.state.copy()
    
    def set_state(self, state):
        """Set ROV state"""
        self.state = state.copy()
    
    def apply_control(self, control):
        """Set control input"""
        self.control_input = control.copy()
        
    def coriolis_matrix(self, vel):
        """
        Compute Coriolis and centripetal matrix
        
        Parameters:
        vel: velocity vector [u, v, w, p, q, r]
        
        Returns:
        6x6 Coriolis matrix
        """
        u, v, w, p, q, r = vel
        
        # Simplified Coriolis matrix
        C = np.zeros((6, 6))
        
        # Translational part
        C[0, 1] = -self.m * r
        C[0, 2] = self.m * q
        C[1, 0] = self.m * r
        C[1, 2] = -self.m * p
        C[2, 0] = -self.m * q
        C[2, 1] = self.m * p
        
        # Rotational part
        C[3, 4] = -self.Iz * r
        C[3, 5] = self.Iy * q
        C[4, 3] = self.Iz * r
        C[4, 5] = -self.Ix * p
        C[5, 3] = -self.Iy * q
        C[5, 4] = self.Ix * p
        
        return C
    
    def dynamics(self, state, t, disturbance):
        """
        ROV dynamics equations
        
        Parameters:
        state: full state vector (12 elements)
        t: time
        disturbance: external disturbance forces/torques
        
        Returns:
        state_dot: derivative of state
        """
        # Split state
        pos = state[:6]      # Position/orientation
        vel = state[6:]      # Linear/angular velocity
        
        # Compute transformation matrix from body to inertial frame
        # Simplified: identity matrix for small angles
        J = np.eye(6)
        
        # Coriolis matrix
        C = self.coriolis_matrix(vel)
        
        # Damping forces (linear + quadratic)
        D_linear = self.D @ vel
        D_quadratic = self.D_nl @ np.abs(vel) * vel
        D_total = D_linear + D_quadratic
        
        # Control forces
        tau = self.control_input
        
        # Compute acceleration
        # M * a + C * v + D * v = tau + disturbance
        acc = np.linalg.solve(self.M, tau + disturbance - C @ vel - D_total)
        
        # State derivative
        state_dot = np.concatenate([J @ vel, acc])
        
        return state_dot
    
    def integrate(self, dt, disturbance=np.zeros(6)):
        """
        Integrate dynamics one step
        
        Parameters:
        dt: time step
        disturbance: external disturbance vector
        """
        # Use simple Euler integration for speed
        state_dot = self.dynamics(self.state, 0, disturbance)
        self.state = self.state + state_dot * dt
        
        # Ensure small angles are wrapped
        self.state[3:6] = np.mod(self.state[3:6] + np.pi, 2*np.pi) - np.pi
        
    def get_position(self):
        """Return position [x, y, z]"""
        return self.state[:3]
    
    def get_orientation(self):
        """Return orientation [roll, pitch, yaw]"""
        return self.state[3:6]
    
    def get_velocity(self):
        """Return velocity [u, v, w, p, q, r]"""
        return self.state[6:]