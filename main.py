import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
import time

def run_console():
    """Run console version with Lyapunov Stability Analysis"""
    from plant.rov_dynamics import ROVDynamics
    from plant.environment import OceanCurrent
    from controller.l1_adaptive import L1AdaptiveControllerStable as L1AdaptiveController
    
    class ROVSimulation:
        def __init__(self):
            self.rov = ROVDynamics()
            self.current = OceanCurrent(current_speed=0.2, direction=np.pi/4)
            self.controller = L1AdaptiveController()
            self.dt = 0.01
            self.t = np.arange(0, 20.0, self.dt)
            self.adapt_params = []

        def reference_trajectory(self, t_array):
            ref = np.zeros((len(t_array), 6))
            for i, t in enumerate(t_array):
                if t >= 2.0:
                    tau = 0.5
                    t_adj = t - 2.0
                    ref[i, 0] = 5.0 * (1 - np.exp(-t_adj / tau))
            return ref

        def run(self):
            states = []
            controls = []
            errors = []
            
            # --- TAMBAHAN UNTUK LYAPUNOV ---
            lyapunov_V = []
            lyapunov_V_dot = []
            
            # Matriks Massa (M) perkiraan dari rov_dynamics.py untuk perhitungan energi
            M_diag = np.array([120.0, 120.0, 130.0, 15.0, 15.0, 23.0])
            Kp = 10.0 # Gain Kp dari L1AdaptiveControllerStable
            # -------------------------------
            
            ref_traj = self.reference_trajectory(self.t)
            
            for i, t_step in enumerate(self.t):
                ref = ref_traj[i]
                state = self.rov.get_state()
                pos = state[:6]
                vel = state[6:]
                current_vel = self.current.get_current_velocity(pos[:3], t_step)
                
                control, sigma_hat, adapt_state = self.controller.compute_control(ref, pos, vel, current_vel, t_step)
                self.rov.apply_control(control)
                self.rov.integrate(self.dt, current_vel[:6])
                
                states.append(state.copy())
                controls.append(control.copy())
                self.adapt_params.append(adapt_state)
                
                error = ref - pos
                errors.append(error)
                
                # --- PERHITUNGAN LYAPUNOV V(t) & dV/dt ---
                # V = 0.5 * e^T * Kp * e + 0.5 * v^T * M * v
                V = 0.5 * np.sum(Kp * (error**2)) + 0.5 * np.sum(M_diag * (vel**2))
                lyapunov_V.append(V)
                
                # Turunan Numerik dV/dt
                if i > 0:
                    V_dot = (V - lyapunov_V[-2]) / self.dt
                else:
                    V_dot = 0.0
                lyapunov_V_dot.append(V_dot)
                # -----------------------------------------
                
            self.adapt_params = np.array(self.adapt_params)
            return np.array(states), np.array(controls), np.array(errors), np.array(lyapunov_V), np.array(lyapunov_V_dot)

        def get_performance_metrics(self):
            states, controls, errors, _, _ = self.run()
            pos_error = errors[:, :6]
            
            # Simple settling time calculation
            threshold = 0.05
            within_threshold = np.abs(pos_error[:, 0]) < threshold
            settling_time = self.t[-1]
            if np.any(within_threshold):
                first_entry = np.argmax(within_threshold)
                if np.all(within_threshold[first_entry:]):
                    settling_time = self.t[first_entry]

            metrics = {
                'ISE': trapezoid(np.sum(pos_error**2, axis=1), self.t),
                'IAE': trapezoid(np.sum(np.abs(pos_error), axis=1), self.t),
                'ITAE': trapezoid(self.t * np.sum(np.abs(pos_error), axis=1), self.t),
                'max_control': np.max(np.abs(controls)),
                'settling_time': settling_time,
                'steady_state_error': np.abs(pos_error[-100:, 0]).mean() if len(pos_error) > 100 else np.abs(pos_error[:, 0]).mean(),
                'overshoot': max(0, (np.max(states[:, 0]) - 5.0) / 5.0 * 100)
            }
            return metrics

    # Run simulation
    sim = ROVSimulation()
    states, controls, errors, lyapunov_V, lyapunov_V_dot = sim.run()
    
    ref_traj = sim.reference_trajectory(sim.t)
    
    # --- PLOT 1: KINERJA KONTROL ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Position tracking
    axes[0, 0].plot(sim.t, states[:, 0], 'b-', label='Actual', linewidth=2)
    axes[0, 0].plot(sim.t, ref_traj[:, 0], 'r--', label='Reference', linewidth=2)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('X Position (m)')
    axes[0, 0].set_title('ROV Position Tracking')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Tracking error
    error = states[:, 0] - ref_traj[:, 0]
    axes[0, 1].plot(sim.t, error, 'g-', linewidth=1.5)
    axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.5)
    axes[0, 1].axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='±0.1m')
    axes[0, 1].axhline(y=-0.1, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Tracking Error (m)')
    axes[0, 1].set_title('Tracking Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Control effort
    axes[1, 0].plot(sim.t, controls[:, 0], 'm-', linewidth=1.5)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Control Input (N)')
    axes[1, 0].set_title('Control Effort')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Adaptive parameters
    axes[1, 1].plot(sim.t, sim.adapt_params[:, 0], 'c-', linewidth=1.5, label='σ_x')
    axes[1, 1].plot(sim.t, sim.adapt_params[:, 1], 'y-', linewidth=1.5, label='σ_y')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Adaptive Parameter')
    axes[1, 1].set_title('Adaptive Parameters')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # --- PLOT 2: KESTABILAN LYAPUNOV ---
    fig2, ax2 = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot V(t)
    ax2[0].plot(sim.t, lyapunov_V, 'b-', linewidth=2)
    ax2[0].set_title('Lyapunov Function $V(t)$ (Total Error Energy)')
    ax2[0].set_ylabel('$V(t)$')
    ax2[0].grid(True, alpha=0.3)
    ax2[0].axhline(y=0, color='k', linestyle='--', alpha=0.8)
    
    # Plot dV/dt
    ax2[1].plot(sim.t, lyapunov_V_dot, 'r-', linewidth=1.5)
    ax2[1].axhline(y=0, color='k', linestyle='--', alpha=0.8)
    ax2[1].fill_between(sim.t, lyapunov_V_dot, 0, where=(lyapunov_V_dot <= 0), color='green', alpha=0.2, label='Stable Region ($\dot{V} \le 0$)')
    ax2[1].fill_between(sim.t, lyapunov_V_dot, 0, where=(lyapunov_V_dot > 0), color='red', alpha=0.2, label='Transient/Injection')
    ax2[1].set_title('Derivative of Lyapunov Function $\dot{V}(t)$')
    ax2[1].set_xlabel('Time (s)')
    ax2[1].set_ylabel('$\dot{V}(t)$')
    ax2[1].legend()
    ax2[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print metrics
    metrics = sim.get_performance_metrics()
    print("\n" + "="*60)
    print("Performance Metrics:")
    print("="*60)
    print(f"  ISE: {metrics['ISE']:.2f}")
    print(f"  IAE: {metrics['IAE']:.2f}")
    print(f"  ITAE: {metrics['ITAE']:.2f}")
    print(f"  Max Control: {metrics['max_control']:.2f} N")
    print(f"  Settling Time: {metrics['settling_time']:.2f} s")
    print(f"  Steady-state Error: {metrics['steady_state_error']:.3f} m")
    print(f"  Overshoot: {metrics['overshoot']:.1f}%")


def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("ROV L1 Adaptive Control System with GA Auto-Tuning")
    print("="*60)
    print("Memulai antarmuka grafis (GUI)...")
    
    try:
        # Langsung mengimpor dan memanggil GUI tanpa meminta input konsol
        from gui.main_window import run_gui
        run_gui()
    except ImportError as e:
        # Fallback ke console (dengan grafik plot) jika PyQt5 gagal diimpor
        print(f"\n[ERROR] Gagal memuat modul GUI: {e}")
        print("Pastikan PyQt5 sudah terinstall dengan: pip install PyQt5 pyqtgraph")
        print("Sistem secara otomatis dialihkan ke Simulasi Konsol...")
        run_console()


if __name__ == "__main__":
    main()