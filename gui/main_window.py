"""
Main GUI Window for ROV L1 Adaptive Control System
"""

import sys
import numpy as np
import json
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
                             QGroupBox, QTabWidget, QTableWidget, QTableWidgetItem,
                             QTextEdit, QProgressBar, QFileDialog, QMessageBox,
                             QSplitter, QComboBox, QCheckBox, QSlider,
                             QDesktopWidget, QScrollArea)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QColor, QPalette
import pyqtgraph as pg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import threading
import time

from plant.rov_dynamics import ROVDynamics
from plant.environment import OceanCurrent
from controller.l1_adaptive import L1AdaptiveControllerStable as L1AdaptiveController
from optimizer.ga_tuner import GATuner


class SimulationThread(QThread):
    """Thread untuk menjalankan simulasi tanpa memblock GUI"""
    
    # Signals
    simulation_update = pyqtSignal(float, np.ndarray, np.ndarray, np.ndarray)
    simulation_finished = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    status_update = pyqtSignal(str)
    
    def __init__(self, controller_params, current_params, sim_params):
        super().__init__()
        self.controller_params = controller_params
        self.current_params = current_params
        self.sim_params = sim_params
        self.running = True
        self.paused = False
        
    def run(self):
        """Run simulation"""
        try:
            # Initialize components (without passing parameters to ROVDynamics)
            rov = ROVDynamics()  # ROVDynamics doesn't accept parameters
            current = OceanCurrent(**self.current_params)
            controller = L1AdaptiveController(**self.controller_params)
            
            # Simulation parameters
            dt = self.sim_params['dt']
            t_sim = self.sim_params['duration']
            t = np.arange(0, t_sim, dt)
            
            # Storage
            states = []
            controls = []
            adapt_params = []
            errors = []
            
            # Reference trajectory function
            def reference(t):
                if self.sim_params['ref_type'] == 'step':
                    if t < 2.0:
                        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    else:
                        x_ref = self.sim_params['ref_value']
                        return np.array([x_ref, 0.0, 0.0, 0.0, 0.0, 0.0])
                elif self.sim_params['ref_type'] == 'smooth_step':
                    if t < 2.0:
                        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    else:
                        tau = self.sim_params['smooth_tau']
                        t_adj = t - 2.0
                        x_ref = self.sim_params['ref_value'] * (1 - np.exp(-t_adj / tau))
                        return np.array([x_ref, 0.0, 0.0, 0.0, 0.0, 0.0])
                elif self.sim_params['ref_type'] == 'sine_wave':
                    freq = self.sim_params['sine_freq']
                    amp = self.sim_params['sine_amp']
                    x_ref = amp * np.sin(2 * np.pi * freq * t)
                    return np.array([x_ref, 0.0, 0.0, 0.0, 0.0, 0.0])
                else:
                    return np.zeros(6)
            
            # Run simulation
            for i, time_step in enumerate(t):
                if not self.running:
                    break
                    
                while self.paused:
                    time.sleep(0.01)
                
                ref = reference(time_step)
                state = rov.get_state()
                pos = state[:6]
                vel = state[6:]
                current_vel = current.get_current_velocity(pos[:3], time_step)
                
                control, sigma_hat, adapt_state = controller.compute_control(
                    ref, pos, vel, current_vel, time_step
                )
                
                rov.apply_control(control)
                rov.integrate(dt, current_vel[:6])
                
                states.append(state.copy())
                controls.append(control.copy())
                adapt_params.append(adapt_state)
                errors.append(ref - pos)
                
                # Emit update setiap 50 langkah
                if i % 50 == 0:
                    self.simulation_update.emit(time_step, state, control, ref)
            
            # Convert to arrays
            if len(states) > 0:
                states = np.array(states)
                controls = np.array(controls)
                adapt_params = np.array(adapt_params)
                errors = np.array(errors)
                t_array = t[:len(states)]
                
                self.simulation_finished.emit(t_array, states, controls, errors)
            else:
                self.status_update.emit("No data collected")
            
        except Exception as e:
            self.status_update.emit(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def stop(self):
        """Stop simulation"""
        self.running = False
    
    def pause(self):
        """Pause simulation"""
        self.paused = True
    
    def resume(self):
        """Resume simulation"""
        self.paused = False

class OptimizationThread(QThread):
    optimization_finished = pyqtSignal(dict, float, dict)
    optimization_error = pyqtSignal(str)
    status_update = pyqtSignal(str)

    def __init__(self, bounds):
        super().__init__()
        self.bounds = bounds

    def run(self):
        try:
            self.status_update.emit("Running DEAP algorithms, please wait...")
            tuner = GATuner(
                bounds=self.bounds,
                pop_size=20,
                n_generations=10,
                crossover_prob=0.8,
                mutation_prob=0.2
            )
            best_params, best_fitness, history = tuner.optimize()
            self.optimization_finished.emit(best_params, best_fitness, history)
        except Exception as e:
            self.optimization_error.emit(str(e))

class MplCanvas(FigureCanvas):
    """Matplotlib canvas untuk plotting"""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
    def clear(self):
        """Clear the canvas"""
        self.fig.clear()
        self.draw()


class ROVGUI(QMainWindow):
    """Main GUI Window"""
    
    def __init__(self):
        super().__init__()
        self.simulation_thread = None
        self.simulation_data = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("ROV L1 Adaptive Control System")
        
        # Dapatkan resolusi layar user
        screen = QDesktopWidget().availableGeometry()
        screen_width = screen.width()
        screen_height = screen.height()
        
        # Set ukuran window ke 85% dari ukuran layar agar proporsional
        window_width = int(screen_width * 0.85)
        window_height = int(screen_height * 0.85)
        
        # Hitung titik koordinat agar window berada di tengah layar (Center)
        pos_x = int((screen_width - window_width) / 2)
        pos_y = int((screen_height - window_height) / 2)
        
        self.setGeometry(pos_x, pos_y, window_width, window_height)
        
        # Set style (Dengan tambahan styling Scrollbar modern dan perbaikan background)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 5px;
                margin-top: 15px;
                padding-top: 15px;
                color: #fff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLabel {
                color: #fff;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 8px 16px;
                text-align: center;
                text-decoration: none;
                font-size: 12px;
                margin: 4px 2px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton#stopBtn {
                background-color: #f44336;
            }
            QPushButton#stopBtn:hover {
                background-color: #da190b;
            }
            QSpinBox, QDoubleSpinBox, QComboBox {
                padding: 5px;
                border: 1px solid #555;
                border-radius: 3px;
                background-color: #3c3c3c;
                color: #fff;
            }
            QTextEdit {
                background-color: #1e1e1e;
                color: #fff;
                font-family: Consolas, monospace;
                border: 1px solid #555;
                border-radius: 4px;
            }
            QTabWidget::pane {
                border: 1px solid #555;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #3c3c3c;
                color: #fff;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #4CAF50;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                text-align: center;
                color: #fff;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
            
            /* ========================================= */
            /* PERBAIKAN BACKGROUND & SCROLL AREA        */
            /* ========================================= */
            QScrollArea {
                border: none;
                background-color: #2b2b2b;
            }
            QScrollArea > QWidget > QWidget {
                background-color: #2b2b2b; /* Menyamakan background isi scroll */
            }
            
            /* ========================================= */
            /* BEAUTIFY SCROLLBAR (TAMPILAN MODERN)      */
            /* ========================================= */
            QScrollBar:vertical {
                border: none;
                background-color: #2b2b2b;
                width: 10px;
                margin: 0px 0px 0px 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #555;
                min-height: 30px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #777;
            }
            QScrollBar::handle:vertical:pressed {
                background-color: #4CAF50;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px; /* Menghilangkan tombol panah atas/bawah yang kuno */
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background-color: transparent; /* Background track transparan */
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel (controls) yang akan di-scroll
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Simulation control group
        sim_group = QGroupBox("Simulation Control")
        sim_layout = QVBoxLayout()
        
        self.btn_start = QPushButton("Start Simulation")
        self.btn_start.clicked.connect(self.start_simulation)
        
        self.btn_stop = QPushButton("Stop Simulation")
        self.btn_stop.setObjectName("stopBtn")
        self.btn_stop.clicked.connect(self.stop_simulation)
        self.btn_stop.setEnabled(False)
        
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.clicked.connect(self.pause_simulation)
        self.btn_pause.setEnabled(False)
        
        self.btn_optimize = QPushButton("Run GA Optimization")
        self.btn_optimize.clicked.connect(self.run_optimization)
        
        # =========================================================
        # TAMBAHAN FITUR EXPORT
        # =========================================================
        self.btn_export = QPushButton("Export Data (JSON)")
        self.btn_export.clicked.connect(self.export_data_json)
        self.btn_export.setEnabled(False)  # Dinonaktifkan sebelum ada data
        # =========================================================

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        sim_layout.addWidget(self.btn_start)
        sim_layout.addWidget(self.btn_pause)
        sim_layout.addWidget(self.btn_stop)
        sim_layout.addWidget(self.btn_optimize)
        sim_layout.addWidget(self.btn_export)  # <--- Masukkan tombol ke layout
        sim_layout.addWidget(self.progress_bar)
        sim_group.setLayout(sim_layout)
        left_layout.addWidget(sim_group)
        
        # Controller parameters group
        controller_group = QGroupBox("Controller Parameters")
        controller_layout = QVBoxLayout()
        
        # Parameter inputs
        param_grid = QVBoxLayout()
        
        self.omega_c_spin = QDoubleSpinBox()
        self.omega_c_spin.setRange(1, 50)
        self.omega_c_spin.setValue(12)
        self.omega_c_spin.setSingleStep(1)
        param_grid.addWidget(QLabel("ω_c (Filter cutoff):"))
        param_grid.addWidget(self.omega_c_spin)
        
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(10, 1000)
        self.gamma_spin.setValue(300)
        self.gamma_spin.setSingleStep(10)
        param_grid.addWidget(QLabel("γ (Adaptive gain):"))
        param_grid.addWidget(self.gamma_spin)
        
        self.k_spin = QDoubleSpinBox()
        self.k_spin.setRange(1, 30)
        self.k_spin.setValue(10)
        self.k_spin.setSingleStep(1)
        param_grid.addWidget(QLabel("k (Feedback gain):"))
        param_grid.addWidget(self.k_spin)
        
        self.A_m_spin = QDoubleSpinBox()
        self.A_m_spin.setRange(-50, -1)
        self.A_m_spin.setValue(-15)
        self.A_m_spin.setSingleStep(1)
        param_grid.addWidget(QLabel("A_m:"))
        param_grid.addWidget(self.A_m_spin)
        
        self.B_m_spin = QDoubleSpinBox()
        self.B_m_spin.setRange(5, 50)
        self.B_m_spin.setValue(15)
        self.B_m_spin.setSingleStep(1)
        param_grid.addWidget(QLabel("B_m:"))
        param_grid.addWidget(self.B_m_spin)
        
        controller_layout.addLayout(param_grid)
        controller_group.setLayout(controller_layout)
        left_layout.addWidget(controller_group)
        
        # Reference parameters group
        ref_group = QGroupBox("Reference Trajectory")
        ref_layout = QVBoxLayout()
        
        self.ref_type_combo = QComboBox()
        self.ref_type_combo.addItems(["Step", "Smooth Step", "Sine Wave"])
        ref_layout.addWidget(QLabel("Reference Type:"))
        ref_layout.addWidget(self.ref_type_combo)
        
        self.ref_value_spin = QDoubleSpinBox()
        self.ref_value_spin.setRange(-10, 10)
        self.ref_value_spin.setValue(5.0)
        self.ref_value_spin.setSingleStep(0.5)
        ref_layout.addWidget(QLabel("Reference Value (m):"))
        ref_layout.addWidget(self.ref_value_spin)
        
        self.smooth_tau_spin = QDoubleSpinBox()
        self.smooth_tau_spin.setRange(0.1, 2.0)
        self.smooth_tau_spin.setValue(0.5)
        self.smooth_tau_spin.setSingleStep(0.1)
        ref_layout.addWidget(QLabel("Smoothing Tau:"))
        ref_layout.addWidget(self.smooth_tau_spin)
        
        self.sine_amp_spin = QDoubleSpinBox()
        self.sine_amp_spin.setRange(0, 5)
        self.sine_amp_spin.setValue(2)
        ref_layout.addWidget(QLabel("Sine Amplitude (m):"))
        ref_layout.addWidget(self.sine_amp_spin)
        
        self.sine_freq_spin = QDoubleSpinBox()
        self.sine_freq_spin.setRange(0.1, 2)
        self.sine_freq_spin.setValue(0.2)
        ref_layout.addWidget(QLabel("Sine Frequency (Hz):"))
        ref_layout.addWidget(self.sine_freq_spin)
        
        ref_group.setLayout(ref_layout)
        left_layout.addWidget(ref_group)
        
        # Simulation parameters group
        sim_param_group = QGroupBox("Simulation Parameters")
        sim_param_layout = QVBoxLayout()
        
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(5, 60)
        self.duration_spin.setValue(20)
        sim_param_layout.addWidget(QLabel("Duration (s):"))
        sim_param_layout.addWidget(self.duration_spin)
        
        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(0.001, 0.05)
        self.dt_spin.setValue(0.01)
        self.dt_spin.setSingleStep(0.005)
        sim_param_layout.addWidget(QLabel("Time Step (s):"))
        sim_param_layout.addWidget(self.dt_spin)
        
        sim_param_group.setLayout(sim_param_layout)
        left_layout.addWidget(sim_param_group)
        
        # Environment parameters group
        env_group = QGroupBox("Environment")
        env_layout = QVBoxLayout()
        
        self.current_speed_spin = QDoubleSpinBox()
        self.current_speed_spin.setRange(0, 1)
        self.current_speed_spin.setValue(0.2)
        self.current_speed_spin.setSingleStep(0.05)
        env_layout.addWidget(QLabel("Current Speed (m/s):"))
        env_layout.addWidget(self.current_speed_spin)
        
        self.current_dir_spin = QDoubleSpinBox()
        self.current_dir_spin.setRange(0, 360)
        self.current_dir_spin.setValue(45)
        env_layout.addWidget(QLabel("Current Direction (deg):"))
        env_layout.addWidget(self.current_dir_spin)
        
        env_group.setLayout(env_layout)
        left_layout.addWidget(env_group)
        
        # Status log
        log_group = QGroupBox("Status Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        left_layout.addWidget(log_group)
        
        left_layout.addStretch()

        # =========================================================
        # PERUBAHAN SCROLL AREA: Membungkus left_panel ke dalam scroll
        # =========================================================
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(left_panel)
        
        # Pindahkan aturan lebar minimum/maksimum ke area scroll
        left_panel_width = int(window_width * 0.25) 
        scroll_area.setMinimumWidth(320) # Beri ruang lebih untuk scrollbar
        scroll_area.setMaximumWidth(int(window_width * 0.35))
        # =========================================================

        # Right panel (plots)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Tab widget for different plots
        self.tab_widget = QTabWidget()
        
        # Position plot
        self.position_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.tab_widget.addTab(self.position_canvas, "Position")
        
        # Velocity plot
        self.velocity_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.tab_widget.addTab(self.velocity_canvas, "Velocity")
        
        # Control plot
        self.control_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.tab_widget.addTab(self.control_canvas, "Control Input")
        
        # Error plot
        self.error_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.tab_widget.addTab(self.error_canvas, "Tracking Error")
        
        # Adaptive parameters plot
        self.adaptive_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.tab_widget.addTab(self.adaptive_canvas, "Adaptive Parameters")
        
        # Metrics table
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.setMaximumHeight(200)
        
        right_layout.addWidget(self.tab_widget)
        right_layout.addWidget(self.metrics_table)
        
        # Add panels to main layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(scroll_area)  # <--- Masukkan scroll_area, BUKAN left_panel
        splitter.addWidget(right_panel)
        
        # Set ukuran rasio splitter secara dinamis
        splitter.setSizes([left_panel_width, window_width - left_panel_width])
        
        main_layout.addWidget(splitter)
        
        self.log("ROV Control System initialized")
        
    def log(self, message):
        """Add message to log"""
        self.log_text.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        
    def start_simulation(self):
        """Start simulation"""
        # Get parameters
        controller_params = {
            'omega_c': self.omega_c_spin.value(),
            'gamma': self.gamma_spin.value(),
            'k': self.k_spin.value(),
            'A_m': self.A_m_spin.value(),
            'B_m': self.B_m_spin.value()
        }
        
        current_params = {
            'current_speed': self.current_speed_spin.value(),
            'direction': np.radians(self.current_dir_spin.value()),
            'depth_factor': 0.1
        }
        
        # Reference type
        ref_type = self.ref_type_combo.currentText().lower().replace(" ", "_")
        
        sim_params = {
            'dt': self.dt_spin.value(),
            'duration': self.duration_spin.value(),
            'ref_type': ref_type,
            'ref_value': self.ref_value_spin.value(),
            'smooth_tau': self.smooth_tau_spin.value(),
            'sine_amp': self.sine_amp_spin.value(),
            'sine_freq': self.sine_freq_spin.value()
        }
        
        # Create and start simulation thread
        self.simulation_thread = SimulationThread(
            controller_params, current_params, sim_params
        )
        self.simulation_thread.simulation_update.connect(self.update_plots_live)
        self.simulation_thread.simulation_finished.connect(self.simulation_finished)
        self.simulation_thread.status_update.connect(self.log)
        self.simulation_thread.start()
        
        # Update button states
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_pause.setEnabled(True)
        
        self.log("Simulation started")
        
    def stop_simulation(self):
        """Stop simulation"""
        if self.simulation_thread:
            self.simulation_thread.stop()
            self.simulation_thread = None
        
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("Pause")
        
        self.log("Simulation stopped")
        
    def pause_simulation(self):
        """Pause/resume simulation"""
        if self.simulation_thread:
            if self.simulation_thread.paused:
                self.simulation_thread.resume()
                self.btn_pause.setText("Pause")
                self.log("Simulation resumed")
            else:
                self.simulation_thread.pause()
                self.btn_pause.setText("Resume")
                self.log("Simulation paused")
    
    def update_plots_live(self, t, state, control, ref):
        """Update plots with live data"""
        # This can be implemented for real-time plotting
        pass
        
    def simulation_finished(self, t, states, controls, errors):
        """Handle simulation completion"""
        self.simulation_data = (t, states, controls, errors)
        
        # Update plots
        self.update_position_plot(t, states, controls, errors)
        self.update_velocity_plot(t, states)
        self.update_control_plot(t, controls)
        self.update_error_plot(t, errors)
        self.update_adaptive_plot(t, controls, states)  # Pass controls for adaptive params
        self.update_metrics_table(states, controls, errors, t)
        
        # Update button states
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("Pause")
        self.btn_export.setEnabled(True)
        
        self.log("Simulation completed")
        
    def update_position_plot(self, t, states, controls, errors):
        """Update position plot"""
        self.position_canvas.clear()
        ax = self.position_canvas.fig.add_subplot(111)
        
        # Get reference trajectory based on current settings
        ref_type = self.ref_type_combo.currentText().lower().replace(" ", "_")
        ref_value = self.ref_value_spin.value()
        
        if ref_type == "step":
            ref = np.where(t < 2.0, 0, ref_value)
        elif ref_type == "smooth_step":
            tau = self.smooth_tau_spin.value()
            ref = np.zeros_like(t)
            for i, ti in enumerate(t):
                if ti < 2.0:
                    ref[i] = 0
                else:
                    t_adj = ti - 2.0
                    ref[i] = ref_value * (1 - np.exp(-t_adj / tau))
        elif ref_type == "sine_wave":
            freq = self.sine_freq_spin.value()
            amp = self.sine_amp_spin.value()
            ref = amp * np.sin(2 * np.pi * freq * t)
        else:
            ref = np.zeros_like(t)
        
        ax.plot(t, states[:, 0], 'b-', linewidth=2, label='Actual X')
        ax.plot(t, ref, 'r--', linewidth=2, label='Reference X')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (m)')
        ax.set_title('ROV Position Tracking')
        ax.legend()
        ax.grid(True, alpha=0.3)
        self.position_canvas.draw()
        
    def update_velocity_plot(self, t, states):
        """Update velocity plot"""
        self.velocity_canvas.clear()
        ax = self.velocity_canvas.fig.add_subplot(111)
        
        if len(states) > 0:
            ax.plot(t, states[:, 6], 'b-', linewidth=2, label='u (surge)')
            ax.plot(t, states[:, 7], 'g-', linewidth=2, label='v (sway)')
            ax.plot(t, states[:, 8], 'r-', linewidth=2, label='w (heave)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title('ROV Velocities')
        ax.legend()
        ax.grid(True, alpha=0.3)
        self.velocity_canvas.draw()
        
    def update_control_plot(self, t, controls):
        """Update control plot"""
        self.control_canvas.clear()
        ax = self.control_canvas.fig.add_subplot(111)
        
        if len(controls) > 0:
            ax.plot(t, controls[:, 0], 'b-', linewidth=2, label='Fx')
            ax.plot(t, controls[:, 1], 'g-', linewidth=2, label='Fy')
            ax.plot(t, controls[:, 2], 'r-', linewidth=2, label='Fz')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Force (N)')
        ax.set_title('Control Forces')
        ax.legend()
        ax.grid(True, alpha=0.3)
        self.control_canvas.draw()
        
    def update_error_plot(self, t, errors):
        """Update tracking error plot"""
        self.error_canvas.clear()
        ax = self.error_canvas.fig.add_subplot(111)
        
        if len(errors) > 0:
            ax.plot(t, errors[:, 0], 'b-', linewidth=2, label='X Error')
            ax.plot(t, errors[:, 1], 'g-', linewidth=2, label='Y Error')
            ax.plot(t, errors[:, 2], 'r-', linewidth=2, label='Z Error')
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error (m)')
        ax.set_title('Tracking Errors')
        ax.legend()
        ax.grid(True, alpha=0.3)
        self.error_canvas.draw()
        
    def update_adaptive_plot(self, t, controls, states):
        """Update adaptive parameters plot"""
        self.adaptive_canvas.clear()
        ax = self.adaptive_canvas.fig.add_subplot(111)
        
        # Show adaptive parameter information from controller
        ax.text(0.5, 0.5, 'Adaptive parameters are updated during simulation\n'
                'They represent the estimated uncertainties\n'
                'and disturbances in the system', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, style='italic')
        ax.set_title('Adaptive Parameters')
        ax.axis('off')
        self.adaptive_canvas.draw()
        
    def update_metrics_table(self, states, controls, errors, t):
        """Update metrics table"""
        if len(states) == 0:
            return
            
        # Calculate metrics
        ref_type = self.ref_type_combo.currentText().lower().replace(" ", "_")
        ref_value = self.ref_value_spin.value()
        
        if ref_type == "step" or ref_type == "smooth_step":
            ref = np.where(t < 2.0, 0, ref_value)
            pos_error = states[:, 0] - ref
        elif ref_type == "sine_wave":
            freq = self.sine_freq_spin.value()
            amp = self.sine_amp_spin.value()
            ref = amp * np.sin(2 * np.pi * freq * t)
            pos_error = states[:, 0] - ref
        else:
            pos_error = errors[:, 0]
        
        # Calculate metrics
        from scipy.integrate import trapezoid
        ise = trapezoid(pos_error**2, t) if len(t) > 0 else 0
        iae = trapezoid(np.abs(pos_error), t) if len(t) > 0 else 0
        itae = trapezoid(t * np.abs(pos_error), t) if len(t) > 0 else 0
        max_control = np.max(np.abs(controls[:, 0])) if len(controls) > 0 else 0
        steady_state_error = np.abs(pos_error[-100:]).mean() if len(pos_error) > 100 else np.abs(pos_error).mean()
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean(pos_error**2))
        
        # Calculate overshoot
        if ref_type in ["step", "smooth_step"]:
            step_start = np.argmax(t >= 2.0)
            if step_start < len(states):
                after_step = states[step_start:, 0]
                max_pos = np.max(after_step)
                overshoot = max(0, (max_pos - ref_value) / abs(ref_value) * 100) if ref_value != 0 else 0
            else:
                overshoot = 0
        else:
            overshoot = 0
        
        # Calculate settling time (2% band)
        settling_time = t[-1] if len(t) > 0 else 0
        if ref_type in ["step", "smooth_step"]:
            threshold = 0.02 * abs(ref_value) if ref_value != 0 else 0.05
            for i in range(len(t)):
                if np.all(np.abs(pos_error[i:]) < threshold):
                    settling_time = t[i]
                    break
        
        # Update table
        self.metrics_table.setRowCount(8)
        metrics = [
            ("ISE", f"{ise:.4f}"),
            ("IAE", f"{iae:.4f}"),
            ("ITAE", f"{itae:.4f}"),
            ("RMSE", f"{rmse:.4f} m"),
            ("Max Control", f"{max_control:.2f} N"),
            ("Steady-state Error", f"{steady_state_error:.4f} m"),
            ("Overshoot", f"{overshoot:.2f}%"),
            ("Settling Time", f"{settling_time:.2f} s")
        ]
        
        for i, (metric, value) in enumerate(metrics):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(metric))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(value))
        
        self.metrics_table.resizeColumnsToContents()
        
    def run_optimization(self):
        """Run GA optimization using QThread"""
        self.log("Starting GA optimization...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.btn_optimize.setEnabled(False)
        
        bounds = {
            'omega_c': (5.0, 50.0),
            'gamma': (100.0, 1000.0),
            'k': (5.0, 30.0),
            'A_m': (-50.0, -5.0),
            'B_m': (5.0, 50.0)
        }
        
        self.opt_thread = OptimizationThread(bounds)
        self.opt_thread.optimization_finished.connect(self.on_opt_finished)
        self.opt_thread.optimization_error.connect(self.on_opt_error)
        self.opt_thread.status_update.connect(self.log)
        self.opt_thread.start()

    def on_opt_finished(self, best_params, best_fitness, history):
        # Update UI with best parameters
        self.omega_c_spin.setValue(best_params['omega_c'])
        self.gamma_spin.setValue(best_params['gamma'])
        self.k_spin.setValue(best_params['k'])
        self.A_m_spin.setValue(best_params['A_m'])
        self.B_m_spin.setValue(best_params['B_m'])
        
        self.log(f"Optimization completed! Best fitness: {best_fitness:.4f}")
        self.log(f"Best parameters: ω_c={best_params['omega_c']:.2f}, "
                f"γ={best_params['gamma']:.2f}, k={best_params['k']:.2f}")
        
        self.progress_bar.setVisible(False)
        self.btn_optimize.setEnabled(True)

    def on_opt_error(self, error_str):
        self.log(f"Optimization error: {error_str}")
        self.progress_bar.setVisible(False)
        self.btn_optimize.setEnabled(True)
    
    def export_data_json(self):
        """Export simulation data to a JSON file"""
        if self.simulation_data is None:
            QMessageBox.warning(self, "Export Error", "Belum ada data simulasi yang tersedia.")
            return
            
        # Unpack data
        t, states, controls, errors = self.simulation_data
        
        # Buka dialog untuk menyimpan file
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self, 
            "Simpan Data Simulasi", 
            "simulation_results.json", 
            "JSON Files (*.json);;All Files (*)", 
            options=options
        )
        
        if file_name:
            # Pastikan ekstensi .json
            if not file_name.endswith('.json'):
                file_name += '.json'
                
            try:
                self.log("Sedang mengekspor data ke JSON...")
                # Numpy arrays tidak bisa di-serialize ke JSON secara langsung, 
                # harus diubah ke Python list menggunakan .tolist()
                export_dict = {
                    "time": t.tolist(),
                    "states": states.tolist(),
                    "controls": controls.tolist(),
                    "tracking_errors": errors.tolist(),
                    "metrics": {
                        # Kita bisa simpan metadata yang mungkin berguna
                        "description": "ROV L1 Adaptive Control Simulation Data",
                        "num_samples": len(t)
                    }
                }
                
                # Tulis ke file JSON
                with open(file_name, 'w') as f:
                    json.dump(export_dict, f, indent=4)
                    
                self.log(f"Data berhasil diekspor ke: {file_name}")
                QMessageBox.information(self, "Export Sukses", "Data simulasi berhasil disimpan dalam format JSON.")
                
            except Exception as e:
                self.log(f"Gagal mengekspor data: {str(e)}")
                QMessageBox.critical(self, "Export Error", f"Terjadi kesalahan saat menyimpan data:\n{str(e)}")


def run_gui():
    """Run the GUI application"""
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = ROVGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_gui()