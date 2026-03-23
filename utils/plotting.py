"""
Plotting utilities for ROV simulation results
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class Plotter:
    """Class for creating various plots from simulation data"""
    
    def __init__(self, style='seaborn-v0_8-darkgrid'):
        """Initialize plotter with style"""
        try:
            plt.style.use(style)
        except:
            pass
        
        self.fig_size = (12, 8)
        
    def plot_trajectory(self, t, states, ref_traj=None):
        """
        Plot ROV trajectory in 3D
        
        Parameters:
        t: time vector
        states: state history [n_samples, 12]
        ref_traj: reference trajectory (optional)
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot actual trajectory
        ax.plot3D(states[:, 0], states[:, 1], states[:, 2], 
                  'b-', linewidth=2, label='Actual Trajectory')
        
        # Plot reference if provided
        if ref_traj is not None:
            ax.plot3D(ref_traj[:, 0], ref_traj[:, 1], ref_traj[:, 2],
                     'r--', linewidth=2, label='Reference')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('ROV 3D Trajectory')
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def plot_position(self, t, states, ref_traj=None):
        """
        Plot position vs time for each DOF
        
        Parameters:
        t: time vector
        states: state history [n_samples, 12]
        ref_traj: reference trajectory (optional)
        """
        fig, axes = plt.subplots(2, 3, figsize=self.fig_size)
        axes = axes.flatten()
        
        labels = ['X (m)', 'Y (m)', 'Z (m)', 'Roll (rad)', 'Pitch (rad)', 'Yaw (rad)']
        
        for i in range(6):
            axes[i].plot(t, states[:, i], 'b-', linewidth=1.5, label='Actual')
            if ref_traj is not None:
                axes[i].plot(t, ref_traj[:, i], 'r--', linewidth=1.5, label='Reference')
            
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel(labels[i])
            axes[i].set_title(f'{labels[i]} vs Time')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
        
        plt.tight_layout()
        return fig
    
    def plot_velocity(self, t, states):
        """
        Plot velocity vs time
        
        Parameters:
        t: time vector
        states: state history [n_samples, 12]
        """
        fig, axes = plt.subplots(2, 3, figsize=self.fig_size)
        axes = axes.flatten()
        
        labels = ['u (m/s)', 'v (m/s)', 'w (m/s)', 
                  'p (rad/s)', 'q (rad/s)', 'r (rad/s)']
        
        for i in range(6):
            axes[i].plot(t, states[:, 6 + i], 'g-', linewidth=1.5)
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel(labels[i])
            axes[i].set_title(f'{labels[i]} vs Time')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_control(self, t, controls):
        """
        Plot control inputs vs time
        
        Parameters:
        t: time vector
        controls: control history [n_samples, 6]
        """
        fig, axes = plt.subplots(2, 3, figsize=self.fig_size)
        axes = axes.flatten()
        
        labels = ['Fx (N)', 'Fy (N)', 'Fz (N)', 
                  'τ_roll (Nm)', 'τ_pitch (Nm)', 'τ_yaw (Nm)']
        
        for i in range(6):
            axes[i].plot(t, controls[:, i], 'm-', linewidth=1.5)
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel(labels[i])
            axes[i].set_title(f'{labels[i]} vs Time')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_error(self, t, error):
        """
        Plot tracking error vs time
        
        Parameters:
        t: time vector
        error: tracking error [n_samples, 6]
        """
        fig, axes = plt.subplots(2, 3, figsize=self.fig_size)
        axes = axes.flatten()
        
        labels = ['X Error (m)', 'Y Error (m)', 'Z Error (m)', 
                  'Roll Error (rad)', 'Pitch Error (rad)', 'Yaw Error (rad)']
        
        for i in range(6):
            axes[i].plot(t, error[:, i], 'r-', linewidth=1.5)
            axes[i].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel(labels[i])
            axes[i].set_title(f'{labels[i]} vs Time')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_adaptive_params(self, t, adapt_params):
        """
        Plot adaptive parameters vs time
        
        Parameters:
        t: time vector
        adapt_params: adaptive parameters [n_samples, 6]
        """
        fig, axes = plt.subplots(2, 3, figsize=self.fig_size)
        axes = axes.flatten()
        
        labels = ['σ_x', 'σ_y', 'σ_z', 'σ_roll', 'σ_pitch', 'σ_yaw']
        
        for i in range(6):
            axes[i].plot(t, adapt_params[:, i], 'c-', linewidth=1.5)
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel(labels[i])
            axes[i].set_title(f'{labels[i]} vs Time')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_all_results(self, t, states, controls, error, ref_traj=None):
        """
        Plot all results in a single figure
        
        Parameters:
        t: time vector
        states: state history
        controls: control history
        error: tracking error
        ref_traj: reference trajectory (optional)
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Position plot
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(t, states[:, 0], 'b-', linewidth=1.5, label='X')
        ax1.plot(t, states[:, 1], 'g-', linewidth=1.5, label='Y')
        ax1.plot(t, states[:, 2], 'r-', linewidth=1.5, label='Z')
        if ref_traj is not None:
            ax1.plot(t, ref_traj[:, 0], 'b--', alpha=0.5, label='X_ref')
            ax1.plot(t, ref_traj[:, 1], 'g--', alpha=0.5, label='Y_ref')
            ax1.plot(t, ref_traj[:, 2], 'r--', alpha=0.5, label='Z_ref')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position (m)')
        ax1.set_title('ROV Position Tracking')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Velocity plot
        ax2 = fig.add_subplot(gs[1, :2])
        ax2.plot(t, states[:, 6], 'b-', linewidth=1.5, label='u')
        ax2.plot(t, states[:, 7], 'g-', linewidth=1.5, label='v')
        ax2.plot(t, states[:, 8], 'r-', linewidth=1.5, label='w')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.set_title('ROV Velocities')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Control plot
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.plot(t, controls[:, 0], 'b-', linewidth=1.5, label='Fx')
        ax3.plot(t, controls[:, 1], 'g-', linewidth=1.5, label='Fy')
        ax3.plot(t, controls[:, 2], 'r-', linewidth=1.5, label='Fz')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Force (N)')
        ax3.set_title('Control Forces')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Tracking error plot
        ax4 = fig.add_subplot(gs[2, :])
        ax4.plot(t, error[:, 0], 'b-', linewidth=1.5, label='X Error')
        ax4.plot(t, error[:, 1], 'g-', linewidth=1.5, label='Y Error')
        ax4.plot(t, error[:, 2], 'r-', linewidth=1.5, label='Z Error')
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Position Error (m)')
        ax4.set_title('Tracking Errors')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('ROV L1 Adaptive Control Results', fontsize=14, fontweight='bold')
        
        return fig
    
    def plot_comparison(self, t, data1, data2, title, ylabel):
        """
        Plot comparison between two controllers
        
        Parameters:
        t: time vector
        data1: data from controller 1
        data2: data from controller 2
        title: plot title
        ylabel: y-axis label
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(t, data1, 'b-', linewidth=1.5, label='Default')
        ax.plot(t, data2, 'r-', linewidth=1.5, label='Optimized')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_convergence(self, history):
        """
        Plot GA convergence history
        
        Parameters:
        history: dictionary with 'min', 'avg', 'std' lists
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        generations = range(1, len(history['min']) + 1)
        
        ax.plot(generations, history['min'], 'b-', linewidth=2, label='Best Fitness')
        ax.plot(generations, history['avg'], 'r-', linewidth=1.5, label='Average Fitness')
        ax.fill_between(generations, 
                        np.array(history['avg']) - np.array(history['std']),
                        np.array(history['avg']) + np.array(history['std']),
                        alpha=0.2, color='gray', label='±1 Std Dev')
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness Value')
        ax.set_title('GA Optimization Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig