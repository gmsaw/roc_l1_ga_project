# 🚀 Design of L1 Adaptive Control System for Remotely Operated Vehicle (ROV)  
## Based on Genetic Algorithm Optimization and Lyapunov Stability Analysis

A comprehensive Python-based simulation framework for underwater **Remotely Operated Vehicles (ROV)**.  
This project implements an **L1 Adaptive Control System**, mathematically validated using **Lyapunov Stability Analysis** and optimized via a **Genetic Algorithm (GA)**.

---

## 🎯 Problem Formulation

This project addresses the following research questions:

1. **How does the L1 Adaptive Control system perform** in maintaining trajectory tracking of an ROV under dynamic uncertainties and ocean disturbances?

2. **How can a Genetic Algorithm (GA) be applied** to optimize and auto-tune L1 adaptive control parameters such as:
   - $\omega_c$ (filter bandwidth)  
   - $\gamma$ (adaptation gain)  
   - $k$ (feedback gain)  

   in order to minimize performance indices (ITAE / ISE)?

3. **How can asymptotic stability be proven** using Lyapunov-based energy functions through computational evaluation?

---

## 🌟 Key Features

| Feature | Description |
|--------|------------|
| **6-DOF ROV Dynamics** | Realistic physical modeling based on Fossen equations (mass, Coriolis, drag) |
| **L1 Adaptive Controller** | Robust control against uncertainties and disturbances |
| **Genetic Algorithm Optimization** | Automatic parameter tuning using DEAP |
| **Lyapunov Stability Analysis** | Real-time energy-based stability validation |
| **Interactive GUI (PyQt5)** | Live visualization and control interface |
| **JSON Export** | Export simulation data for further analysis |

---

## 📁 Project Structure

```
roc_l1_ga_project/
│
├── controller/
│   └── l1_adaptive.py
│
├── gui/
│   └── main_window.py
│
├── optimizer/
│   ├── ga_tuner.py
│   └── plotting.py
│
├── plant/
│   ├── environment.py
│   └── rov_dynamics.py
│
├── utils/
│   └── plotting.py
│
├── main.py
└── requirements.txt
```

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/gmsaw/roc_l1_ga_project.git
cd roc_l1_ga_project
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Linux / MacOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

Run the simulation and launch the GUI:

```bash
python main.py
```

---

## 🖥️ GUI Guide

| Component | Function |
|----------|--------|
| Simulation Control | Start / Pause / Stop simulation |
| Parameter Tuning | Manual tuning or GA optimization |
| Trajectory | Step, Smooth Step, Sine Wave |
| Environment | Simulate ocean disturbances |
| Lyapunov Tab | Stability analysis visualization |
| Export JSON | Save simulation data |

---

## 🧮 Mathematical Foundation

### 1. Lyapunov Stability Function

The system energy function is defined as:

$$
V(t) = \frac{1}{2} e(t)^T K_p e(t) + \frac{1}{2} v(t)^T M v(t)
$$

Where:

- **e(t)** = tracking error  
- **v(t)** = velocity  
- **K_p** = control gain  
- **M** = mass matrix  

The system is asymptotically stable if:

$$
\frac{dV(t)}{dt} \leq 0
$$

---

### 2. Adaptation Law (L1 Adaptive Control)

$$
\frac{d\sigma}{dt} = \Gamma \cdot \text{Proj}(\sigma, -\tilde{x})
$$

Where:

- **$\Gamma$** = adaptation gain  
- **$\tilde{x}$** = state estimation error  
- **Proj** = projection operator  

---

## 📊 Performance Metrics

| Metric | Formula | Description |
|--------|--------|------------|
| ITAE | $ \int_0^T t |e(t)| dt $ | Penalizes long-duration error |
| ISE | $ \int_0^T |e(t)|^2 dt $ | Penalizes large errors |
| IAE | $ \int_0^T |e(t)| dt $ | Total accumulated error |

---

## 📄 Publication (In Progress)

This project is being developed into an IEEE-style research paper covering:

- L1 Adaptive Control Design  
- Genetic Algorithm Optimization  
- Lyapunov Stability Analysis  

---

## 👤 Author

**Gede Mahendra Sastra Adhi Wiguna**  
Physics Department, Universitas Udayana  