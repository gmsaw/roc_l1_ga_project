"""
L1 Adaptive Controller for ROV Position Control
Implements state predictor, adaptive law, and low-pass filter
"""

import numpy as np
from scipy.integrate import odeint


class L1AdaptiveController:
    """
    L1 Adaptive Controller for 6-DOF ROV
    
    The controller consists of:
    1. State predictor: estimates system states
    2. Adaptive law: estimates unknown parameters and disturbances
    3. Control law: generates control input with low-pass filter
    """
    
    def __init__(self, omega_c=15.0, gamma=500.0, k=12.0, A_m=-20.0, B_m=20.0):
        """
        Initialize L1 adaptive controller
        
        Parameters:
        omega_c: low-pass filter cutoff frequency (rad/s)
        gamma: adaptive gain (positive) - REDUCED for stability
        k: feedback gain
        A_m: desired system matrix
        B_m: desired input matrix
        """
        self.omega_c = omega_c
        self.gamma = gamma
        self.k = k
        self.A_m = A_m
        self.B_m = B_m
        
        # Low-pass filter coefficients
        self.filter_alpha = omega_c
        self.filter_beta = omega_c
        
        # State predictor variables
        self.x_hat = np.zeros(6)      # Predicted position states
        self.sigma_hat = np.zeros(6)  # Adaptive parameter estimates
        self.sigma_dot = np.zeros(6)  # Derivative of adaptive parameters
        
        # Filter states
        self.filter_state = np.zeros(6)
        
        # Error tracking
        self.error = np.zeros(6)
        self.integral_error = np.zeros(6)
        
        # Control input
        self.u = np.zeros(6)
        
        # Time step
        self.dt = 0.01
        
        # Saturation limits
        self.sigma_max = 50.0  # Max adaptive parameter magnitude
        
    def reset(self):
        """Reset controller states"""
        self.x_hat = np.zeros(6)
        self.sigma_hat = np.zeros(6)
        self.sigma_dot = np.zeros(6)
        self.filter_state = np.zeros(6)
        self.error = np.zeros(6)
        self.integral_error = np.zeros(6)
        self.u = np.zeros(6)
        
    def state_predictor(self, x, u, sigma_hat):
        """
        State predictor dynamics
        
        Parameters:
        x: actual state (position)
        u: control input
        sigma_hat: adaptive parameter estimate
        
        Returns:
        x_hat_dot: derivative of predicted state
        """
        # Simplified predictor dynamics with stability bounds
        x_hat_dot = self.A_m * self.x_hat + self.B_m * (u + sigma_hat)
        
        # Add correction term with saturation to prevent instability
        error = x - self.x_hat
        correction_gain = 10.0
        correction = correction_gain * np.clip(error, -5.0, 5.0)
        x_hat_dot = x_hat_dot + correction
        
        return x_hat_dot
    
    def adaptive_law(self, x, x_hat):
        """
        Adaptive law for parameter estimation using projection operator
        
        Parameters:
        x: actual state
        x_hat: predicted state
        
        Returns:
        sigma_dot: derivative of adaptive parameters
        """
        # Tracking error
        e = x - x_hat
        
        # Clip error to prevent overflow
        e_clipped = np.clip(e, -10.0, 10.0)
        
        # Simple adaptive law with projection
        sigma_dot = self.gamma * e_clipped
        
        # Projection operator to keep parameters bounded
        for i in range(6):
            # Check if parameter is at boundary and trying to increase
            if np.abs(self.sigma_hat[i]) >= self.sigma_max:
                if sigma_dot[i] * self.sigma_hat[i] > 0:
                    sigma_dot[i] = 0
        
        # Limit rate of change
        sigma_dot = np.clip(sigma_dot, -100.0, 100.0)
        
        return sigma_dot
    
    def low_pass_filter(self, sigma_hat):
        """
        Low-pass filter for control signal
        
        Parameters:
        sigma_hat: adaptive parameter estimate
        
        Returns:
        filtered_sigma: filtered parameter estimate
        """
        # First-order low-pass filter with bounded inputs
        sigma_clipped = np.clip(sigma_hat, -self.sigma_max, self.sigma_max)
        filter_dot = -self.filter_alpha * self.filter_state + self.filter_beta * sigma_clipped
        
        # Limit filter rate
        filter_dot = np.clip(filter_dot, -200.0, 200.0)
        
        # Euler integration with saturation
        self.filter_state = self.filter_state + filter_dot * self.dt
        self.filter_state = np.clip(self.filter_state, -self.sigma_max, self.sigma_max)
        
        return self.filter_state.copy()
    
    def compute_control(self, ref, pos, vel, disturbance, t):
        """
        Compute control input based on L1 adaptive control law
        
        Parameters:
        ref: reference position (6D)
        pos: current position (6D)
        vel: current velocity (6D)
        disturbance: external disturbance (6D)
        t: current time
        
        Returns:
        u: control input (6D)
        sigma_hat: adaptive parameters
        filter_state: filtered adaptive parameters
        """
        # Update time step
        self.dt = 0.01
        
        # Tracking error
        error = ref - pos
        
        # Store error for monitoring
        self.error = error.copy()
        
        # Integral term with anti-windup
        self.integral_error = self.integral_error + error * self.dt
        self.integral_error = np.clip(self.integral_error, -5.0, 5.0)
        
        # State predictor update with bounds
        x_hat_dot = self.state_predictor(pos, self.u, self.sigma_hat)
        self.x_hat = self.x_hat + x_hat_dot * self.dt
        self.x_hat = np.clip(self.x_hat, -20.0, 20.0)
        
        # Adaptive law update
        sigma_dot = self.adaptive_law(pos, self.x_hat)
        self.sigma_hat = self.sigma_hat + sigma_dot * self.dt
        self.sigma_hat = np.clip(self.sigma_hat, -self.sigma_max, self.sigma_max)
        
        # Apply low-pass filter to adaptive parameters
        filtered_sigma = self.low_pass_filter(self.sigma_hat)
        
        # Baseline control law (PD + feedforward)
        Kp = self.k * np.ones(6)
        Kd = 8.0 * np.ones(6)  # Increased damping
        
        # PD control with bounded output
        u_pd = Kp * error + Kd * (-vel)
        u_pd = np.clip(u_pd, -50.0, 50.0)
        
        # Feedforward from reference
        u_ff = self.B_m * error  # Feedforward on error instead of reference
        
        # L1 adaptive component (reduce magnitude for stability)
        u_adapt = -0.8 * filtered_sigma  # Reduced adaptation gain
        
        # Disturbance compensation
        u_disturb = -0.3 * disturbance[:6]  # Partial disturbance rejection
        
        # Total control input
        u = u_pd + u_ff + u_adapt + u_disturb
        
        # Apply control limits
        max_thrust = 80.0  # Reduced max thrust (Newtons)
        max_torque = 40.0   # Reduced max torque (Newton-meters)
        
        u[:3] = np.clip(u[:3], -max_thrust, max_thrust)
        u[3:] = np.clip(u[3:], -max_torque, max_torque)
        
        self.u = u
        
        return u, self.sigma_hat.copy(), filtered_sigma.copy()
    
    def get_parameters(self):
        """Return current adaptive parameters"""
        return self.sigma_hat.copy()
    
    def set_parameters(self, omega_c=None, gamma=None, k=None, A_m=None, B_m=None):
        """Update controller parameters with bounds"""
        if omega_c is not None:
            self.omega_c = max(1.0, min(50.0, omega_c))
            self.filter_alpha = self.omega_c
            self.filter_beta = self.omega_c
            
        if gamma is not None:
            self.gamma = max(10.0, min(1000.0, gamma))  # Bound gamma
            
        if k is not None:
            self.k = max(1.0, min(30.0, k))
            
        if A_m is not None:
            self.A_m = max(-50.0, min(-5.0, A_m))
            
        if B_m is not None:
            self.B_m = max(5.0, min(50.0, B_m))


class L1AdaptiveControllerStable(L1AdaptiveController):
    """
    More stable version of L1 Adaptive Controller with additional safeguards
    """
    
    def __init__(self, omega_c=12.0, gamma=300.0, k=10.0, A_m=-15.0, B_m=15.0):
        super().__init__(omega_c, gamma, k, A_m, B_m)
        self.error_integral = np.zeros(6)
        self.last_error = np.zeros(6)
        
    def compute_control(self, ref, pos, vel, disturbance, t):
        """
        Compute control input with additional stability features
        """
        # Tracking error with filtering
        error = ref - pos
        
        # Apply deadzone to prevent chatter
        deadzone = 0.01
        error_deadzone = np.where(np.abs(error) < deadzone, 0, error)
        
        # Store error
        self.error = error_deadzone.copy()
        
        # PID component (more stable than pure PD)
        Kp = self.k
        Ki = 2.0
        Kd = 6.0
        
        # Proportional
        u_p = Kp * error_deadzone
        
        # Integral (with anti-windup)
        self.error_integral = self.error_integral + error_deadzone * self.dt
        self.error_integral = np.clip(self.error_integral, -3.0, 3.0)
        u_i = Ki * self.error_integral
        
        # Derivative (with filtering)
        error_deriv = (error_deadzone - self.last_error) / self.dt
        error_deriv = np.clip(error_deriv, -10.0, 10.0)
        u_d = Kd * error_deriv - Kd * vel[:6]
        
        # Combined PID
        u_pid = u_p + u_i + u_d
        u_pid = np.clip(u_pid, -60.0, 60.0)
        
        # Adaptive component (with slower adaptation)
        # Only adapt when error is significant
        adapt_gain = 0.5 if np.max(np.abs(error)) > 0.1 else 0.1
        u_adapt = -adapt_gain * self.filter_state
        
        # Disturbance feedforward
        u_dist = -0.2 * disturbance[:6]
        
        # Total control
        u = u_pid + u_adapt + u_dist
        
        # Apply limits
        u[:3] = np.clip(u[:3], -60.0, 60.0)
        u[3:] = np.clip(u[3:], -30.0, 30.0)
        
        self.u = u
        self.last_error = error_deadzone.copy()
        
        # Update adaptive parameters slowly
        if np.max(np.abs(error)) > 0.05:
            sigma_dot = self.gamma * error_deadzone * 0.1  # Slower adaptation
            sigma_dot = np.clip(sigma_dot, -10.0, 10.0)
            self.sigma_hat = self.sigma_hat + sigma_dot * self.dt
            self.sigma_hat = np.clip(self.sigma_hat, -20.0, 20.0)
            
            # Update filter
            filter_dot = -self.filter_alpha * self.filter_state + self.filter_beta * self.sigma_hat
            filter_dot = np.clip(filter_dot, -50.0, 50.0)
            self.filter_state = self.filter_state + filter_dot * self.dt
            self.filter_state = np.clip(self.filter_state, -20.0, 20.0)
        
        return u, self.sigma_hat.copy(), self.filter_state.copy()