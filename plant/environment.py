"""
Ocean current environment model for ROV simulation
Implements realistic current profiles including shear, turbulence, and spatial variations
"""

import numpy as np


class OceanCurrent:
    """Ocean current model for underwater vehicle simulation"""
    
    def __init__(self, current_speed=0.5, direction=np.pi/4, depth_factor=0.1):
        """
        Initialize ocean current model
        
        Parameters:
        current_speed: surface current speed (m/s)
        direction: current direction in radians
        depth_factor: exponential decay factor with depth
        """
        self.current_speed = current_speed
        self.direction = direction
        self.depth_factor = depth_factor
        
        # Current components in inertial frame
        self.current_velocity_xy = current_speed * np.array([
            np.cos(direction),
            np.sin(direction)
        ])
        
        # Turbulence parameters
        self.turbulence_intensity = 0.1  # 10% turbulence intensity
        self.turbulence_frequency = 0.5  # Hz
        
        # Current profile parameters
        self.time = 0.0
        
    def update_time(self, dt):
        """Update internal time for time-varying currents"""
        self.time += dt
        
    def get_current_velocity(self, position, t):
        """
        Get current velocity at given position and time
        
        Parameters:
        position: 3D position [x, y, z] in meters
        t: time in seconds
        
        Returns:
        current_vel: 6D vector [u_current, v_current, w_current, 0, 0, 0]
        (only translational velocities, no rotational)
        """
        x, y, z = position
        
        # Depth-dependent current (exponential decay)
        depth_scale = np.exp(-z * self.depth_factor) if z <= 0 else 1.0
        
        # Base horizontal current with depth variation
        u_horizontal = self.current_velocity_xy[0] * depth_scale
        v_horizontal = self.current_velocity_xy[1] * depth_scale
        
        # Add turbulence (time and position varying)
        turbulence = self._generate_turbulence(x, y, t)
        
        # Vertical current (small upwelling/downwelling)
        w_vertical = 0.05 * np.sin(2 * np.pi * 0.1 * t) * depth_scale
        
        # Combine components
        current_vel = np.array([
            u_horizontal + turbulence[0],
            v_horizontal + turbulence[1],
            w_vertical + turbulence[2],
            0.0,  # No rotational current
            0.0,
            0.0
        ])
        
        return current_vel
    
    def _generate_turbulence(self, x, y, t):
        """
        Generate turbulence components using multiple frequencies
        
        Parameters:
        x, y: position coordinates
        t: time
        
        Returns:
        turbulence: 3D turbulence vector
        """
        # Simple turbulence model using sine waves
        # In reality, this could be replaced with more sophisticated models
        # like the von Karman spectrum or measured data
        
        omega1 = 2 * np.pi * self.turbulence_frequency
        omega2 = 2 * np.pi * self.turbulence_frequency * 1.5
        omega3 = 2 * np.pi * self.turbulence_frequency * 2.0
        
        # Spatial variation
        spatial_scale = 0.01
        kx = spatial_scale * x
        ky = spatial_scale * y
        
        # Turbulence intensity
        intensity = self.turbulence_intensity * self.current_speed
        
        # Generate turbulence components
        turb_x = intensity * (
            np.sin(omega1 * t + kx) + 
            0.5 * np.sin(omega2 * t + ky) +
            0.3 * np.sin(omega3 * t)
        )
        
        turb_y = intensity * (
            np.cos(omega1 * t + ky) + 
            0.5 * np.cos(omega2 * t + kx) +
            0.3 * np.cos(omega3 * t + kx + ky)
        )
        
        turb_z = intensity * 0.3 * (
            np.sin(omega2 * t + kx) * np.cos(omega1 * t + ky)
        )
        
        return np.array([turb_x, turb_y, turb_z])
    
    def get_current_profile(self, z_range):
        """
        Get current velocity profile as function of depth
        
        Parameters:
        z_range: array of depths
        
        Returns:
        profile: current speed at each depth
        """
        profile = []
        for z in z_range:
            depth_scale = np.exp(-z * self.depth_factor) if z <= 0 else 1.0
            speed = self.current_speed * depth_scale
            profile.append(speed)
        
        return np.array(profile)
    
    def set_current_speed(self, speed):
        """Update surface current speed"""
        self.current_speed = speed
        self.current_velocity_xy = speed * np.array([
            np.cos(self.direction),
            np.sin(self.direction)
        ])
    
    def set_direction(self, direction):
        """Update current direction"""
        self.direction = direction
        self.current_velocity_xy = self.current_speed * np.array([
            np.cos(direction),
            np.sin(direction)
        ])
    
    def set_turbulence_intensity(self, intensity):
        """Set turbulence intensity level"""
        self.turbulence_intensity = np.clip(intensity, 0.0, 0.5)


class CurrentProfile:
    """Different current profile types for testing"""
    
    @staticmethod
    def constant(speed, direction):
        """Constant current profile"""
        return OceanCurrent(speed, direction, depth_factor=0.0)
    
    @staticmethod
    def exponential_decay(speed, direction, decay_rate=0.1):
        """Exponentially decaying current with depth"""
        return OceanCurrent(speed, direction, depth_factor=decay_rate)
    
    @staticmethod
    def linear_profile(speed, direction, depth):
        """Linearly varying current with depth"""
        # Not implemented in this version
        pass